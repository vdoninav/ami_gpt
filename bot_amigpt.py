import atexit
import sqlite3
from sys import stderr

import numpy as np
import pymorphy3
import regex as re
import telebot
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import MBartForConditionalGeneration, AutoTokenizer

import amigpt_tokens

np.random.seed(77)
torch.manual_seed(77)

bot_token = amigpt_tokens.BOT_TOKEN
bot_username = amigpt_tokens.BOT_NAME
bot = telebot.TeleBot(bot_token)

model_name = "models/amigpt5"
tok = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
stderr.write(f"Device: {device.type}\n")
model.to(device)

do_summarize = True
if do_summarize:
    model_name_summ = "models/mbart_ruDialogSum"
    tokenizer_summ = AutoTokenizer.from_pretrained(model_name_summ)
    model_summ = MBartForConditionalGeneration.from_pretrained(model_name_summ)
    model_summ.eval()

morph = pymorphy3.MorphAnalyzer()

is_insane = False

MAX_MESSAGE_LENGTH = 4096
max_response_size = 40
min_check_length = 11
max_hist_default = 4

# Подключение к базе данных SQLite и создание таблицы для хранения истории сообщений
conn = sqlite3.connect('message_history.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

cursor.execute('''
CREATE INDEX IF NOT EXISTS idx_user_id ON history (user_id)
''')
conn.commit()

# Счетчик последних сообщений
cursor.execute('''
CREATE TABLE IF NOT EXISTS user_counters (
    user_id INTEGER PRIMARY KEY,
    current_message_count INTEGER DEFAULT 0
)
''')
conn.commit()

# Создание таблицы для хранения максимального лимита сообщений
cursor.execute(f'''
CREATE TABLE IF NOT EXISTS user_limits (
    user_id INTEGER PRIMARY KEY,
    max_messages INTEGER DEFAULT {max_hist_default}
)
''')
conn.commit()


def summarize(text):
    text1 = text
    input_ids = tokenizer_summ(
        [text1],
        max_length=300,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    output_ids = model_summ.generate(
        input_ids=input_ids,
        # top_k=0,
        num_beams=3,
        no_repeat_ngram_size=3
    )[0]

    summary = tokenizer_summ.decode(output_ids, skip_special_tokens=True)

    return summary


def is_russian_word(word, min_length=3):
    # Пропускаем слова, которые меньше заданной длины или содержат только цифры
    if len(word) <= min_length or word.isdigit():
        return True

    # Проверяем каждую часть слова, если оно содержит дефис
    parts = word.split('-')
    return all(any(parse.is_known for parse in morph.parse(part)) for part in parts)


def strip_me(input_string, n=1, min_length=3):
    # Найти позицию n-й новой строки
    pos = -1
    for _ in range(n):
        pos = input_string.find('\n', pos + 1)
        if pos == -1:
            return "DEBUG MESSAGE - не найдена n-я новая строка"

    # Извлекаем подстроку после n-й новой строки
    sub_string = input_string[pos + 1:]

    # Ищем кириллический символ
    cyr_match = re.search(r'\p{IsCyrillic}', sub_string)
    if not cyr_match:
        return "что?"

    # Отрезаем строку от первого кириллического символа
    sub_string = sub_string[cyr_match.start():]

    # Ищем конец сообщения: либо новая строка в пределах 40 символов, либо знаки препинания после 40-го символа
    n = 300
    if len(sub_string) > n:
        punc_positions = [sub_string.find(punc) for punc in '.!?\n' if
                          sub_string.find(punc) != -1 and sub_string.find(punc) < n]
        if punc_positions:  # Если есть найденные символы
            punc_pos = min(punc_positions)
            sub_string = sub_string[:punc_pos + 1].strip()
        else:
            sub_string = sub_string[:n].strip()
    else:
        punc_positions = [sub_string.find(punc) for punc in '.!?' if sub_string.find(punc) != -1]
        if punc_positions:
            punc_pos = min(punc_positions)
            sub_string = sub_string[:punc_pos + 1].strip()

    # Убираем лишние пробелы и фильтруем слова по их валидности
    words = re.split(r'\s+', sub_string)
    filtered_words = [word for word in words if is_russian_word(word, min_length)]
    sub_string = ' '.join(filtered_words)

    # Финальная проверка на наличие кириллических символов
    if not re.search(r'[А-Яа-яЁё]', sub_string):
        return "что?"

    return sub_string


def split_message(message):
    return [message[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(message), MAX_MESSAGE_LENGTH)]


def create_main_keyboard():
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    help_button = telebot.types.KeyboardButton('Help')
    reset_button = telebot.types.KeyboardButton('Reset')
    keyboard.add(help_button, reset_button)
    return keyboard


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! How can I assist you today?", reply_markup=create_main_keyboard())


@bot.message_handler(commands=['reset'])
def reset_history(message):
    user_id = message.from_user.id
    cursor.execute('''
        INSERT OR REPLACE INTO user_counters (user_id, current_message_count)
        VALUES (?, 0)
    ''', (user_id,))
    conn.commit()
    bot.reply_to(message, "Your history has been reset.", reply_markup=create_main_keyboard())


def increment_message_count(user_id):
    cursor.execute('SELECT current_message_count FROM user_counters WHERE user_id = ?', (user_id,))
    current_count = cursor.fetchone()

    if current_count:
        current_count = current_count[0]
        cursor.execute('SELECT max_messages FROM user_limits WHERE user_id = ?', (user_id,))
        max_count = cursor.fetchone()
        max_count = max_count[0] if max_count else max_hist_default  # num_db_msgs по умолчанию

        # Увеличиваем счетчик, но не превышаем максимальное значение
        if current_count < max_count:
            current_count += 1
            cursor.execute('UPDATE user_counters SET current_message_count = ? WHERE user_id = ?',
                           (current_count, user_id))
        conn.commit()
    else:
        cursor.execute('INSERT INTO user_counters (user_id, current_message_count) VALUES (?, ?)', (user_id, 1))
        conn.commit()


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    global is_insane

    if message.chat.type in ['group', 'supergroup']:
        if is_insane:
            response = process_message(user_id, message.text)
            bot.reply_to(message, response, reply_markup=create_main_keyboard())
        if f"@{bot_username}" in message.text:
            response = process_message(user_id, message.text[len(bot_username) + 2:])
            bot.reply_to(message, response, reply_markup=create_main_keyboard())
    else:
        if message.text.lower() == 'help':
            send_welcome(message)
            return
        elif message.text.lower() == 'reset':
            reset_history(message)
            return
        elif amigpt_tokens.INSANITY_ON in message.text:
            is_insane = True
            bot.reply_to(message, "I\'m INSANE!!!", reply_markup=create_main_keyboard())
            return
        elif amigpt_tokens.INSANITY_OFF in message.text:
            is_insane = False
            bot.reply_to(message, "I\'m NOT INSANE.", reply_markup=create_main_keyboard())
            return
        elif "set hist size" in message.text.lower():
            max_history = int(re.search(r'\d+', message.text).group())
            cursor.execute('INSERT OR REPLACE INTO user_limits (user_id, max_messages) VALUES (?, ?)',
                           (user_id, max_history))
            conn.commit()
            bot.reply_to(message, f"Max history size set to {max_history}", reply_markup=create_main_keyboard())
            return

        # Обычное сообщение
        save_message(user_id, f"{message.text}")

        # Увеличиваем счетчик
        increment_message_count(user_id)

        response = process_message(user_id, message.text)
        while response.upper() == response:
            response = process_message(user_id, message.text)

        bot.reply_to(message, response, reply_markup=create_main_keyboard())
        # Сохранение ответа бота в базу данных
        save_message(user_id, f"{response}")

        # Увеличиваем счетчик после ответа нейросети
        increment_message_count(user_id)


def process_message(user_id, text):
    global max_response_size, min_check_length
    cursor.execute('SELECT current_message_count FROM user_counters WHERE user_id = ?', (user_id,))
    count_row = cursor.fetchone()
    message_count = count_row[0] if count_row else 0

    msgs = get_message_history(user_id, message_count)
    dialog_history = "<s>" + "\n<s>".join(msgs) + "\n"
    in_prompt = f"{dialog_history}"

    if do_summarize and len(in_prompt) > 300:
        in_prompt = "<s>" + text + "\n<s>" + summarize(in_prompt)

        current_newline_count = in_prompt.count("\n")
        if current_newline_count < len(msgs):
            in_prompt += "\n" * (len(msgs) - current_newline_count)

    inpt = tok.encode(in_prompt + '\n', return_tensors="pt")
    max_len = max_response_size + len(in_prompt)
    out = model.generate(inpt.to(device), max_length=max_len, repetition_penalty=5.0,
                         do_sample=True, top_k=5, top_p=0.95, temperature=0.7, no_repeat_ngram_size=3)
    response = strip_me(tok.decode(out[0]), len(msgs), min_check_length)

    return response


def save_message(user_id, message, n=10):
    # Сохранение нового сообщения в базу данных
    cursor.execute('INSERT INTO history (user_id, message) VALUES (?, ?)', (user_id, message))
    conn.commit()

    # Подсчет количества сообщений у пользователя
    cursor.execute('SELECT COUNT(*) FROM history WHERE user_id = ?', (user_id,))
    message_count = cursor.fetchone()[0]

    # Если сообщений больше 2n, удаляем самые старые
    # if message_count > 2 * n:
    #     excess_count = message_count - 2 * n
    #     cursor.execute('SELECT id FROM history WHERE user_id = ? ORDER BY timestamp LIMIT ?', (user_id, excess_count))
    #     ids_to_delete = cursor.fetchall()
    #
    #     for row in ids_to_delete:
    #         cursor.execute('DELETE FROM history WHERE id = ?', (row[0],))
    #     conn.commit()


def get_message_history(user_id, n=3):
    cursor.execute('SELECT message FROM history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, n))
    rows = cursor.fetchall()

    # Сообщения будут в обратном порядке, поэтому разворачиваем их перед возвратом
    return [row[0] for row in reversed(rows)]


def clear_user_history(user_id):
    cursor.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
    conn.commit()


def send_response(message, response):
    if response:
        if len(response) > MAX_MESSAGE_LENGTH:
            for chunk in split_message(response):
                bot.reply_to(message, chunk)
        else:
            bot.reply_to(message, response)


stderr.write("Initialized successfully\n")

bot.infinity_polling()

# Закрытие соединения с базой данных при завершении работы бота
atexit.register(lambda: conn.close())
