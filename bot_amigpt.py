import telebot
import numpy as np
import torch
import sqlite3
import time
import atexit
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import amigpt_tokens

np.random.seed(77)
torch.manual_seed(77)

bot_token = amigpt_tokens.BOT_TOKEN
bot_username = amigpt_tokens.BOT_NAME
bot = telebot.TeleBot(bot_token)

tok = GPT2Tokenizer.from_pretrained("models/amigpt_large_2")
model = GPT2LMHeadModel.from_pretrained("models/amigpt_large_2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

is_insane = False

MAX_MESSAGE_LENGTH = 4096

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

import regex as re

import regex as re


def strip_me(input_string, n=1):
    # Найти позицию n-й новой строки
    pos = -1
    for _ in range(n):
        pos = input_string.find('\n', pos + 1)
        if pos == -1:
            return "DEBUG MESSAGE - не найдена n-я новая строка"  # Если n-я новая строка не найдена, возвращаем пустую строку

    # Извлекаем подстроку после n-й новой строки
    sub_string = input_string[pos + 1:]

    # Ищем кириллический символ
    cyr_match = re.search(r'\p{IsCyrillic}', sub_string)

    # Если кириллических символов нет, возвращаем "что?"
    if not cyr_match:
        return "что?"

    # Отрезаем строку от первого кириллического символа
    sub_string = sub_string[cyr_match.start():]

    # Ищем конец сообщения: либо новая строка в пределах 40 символов, либо знаки препинания после 40-го символа
    if len(sub_string) > 40:
        punc_match = re.search(r'[.!?]', sub_string[40:])
        if punc_match:
            sub_string = sub_string[:40 + punc_match.end()].strip()
        else:
            newline_match = re.search(r'\n', sub_string[:40])
            if newline_match:
                sub_string = sub_string[:newline_match.start()].strip()
            else:
                sub_string = sub_string[:40].strip()
    else:
        punc_match = re.search(r'[.!?]', sub_string)
        if punc_match:
            sub_string = sub_string[:punc_match.end()].strip()

    # Убираем лишние пробелы и возвращаем строку
    sub_string = re.sub(r'\s+', ' ', sub_string)

    b = re.search(r'[А-Яа-яЁё]', sub_string)
    if not b:
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
    clear_user_history(user_id)
    bot.reply_to(message, "Your history has been reset.", reply_markup=create_main_keyboard())


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    global is_insane, num_db_msgs

    # Сообщения в группах
    if message.chat.type in ['group', 'supergroup']:
        if is_insane:
            response = process_message(user_id, message.text)
            bot.reply_to(message, response, reply_markup=create_main_keyboard())
        if f"@{bot_username}" in message.text:
            response = process_message(user_id, message.text[len(bot_username) + 2:])
            bot.reply_to(message, response, reply_markup=create_main_keyboard())
    # Сообщения в лс
    else:
        # Обработка команд
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
        elif "set db size" in message.text.lower():
            num_db_msgs = int(re.search(r'\d+', message.text).group())
            bot.reply_to(message, f"DB size set to {num_db_msgs}", reply_markup=create_main_keyboard())
            return

        # Обычное сообщение
        save_message(user_id, f"{message.text}", num_db_msgs)

        response = process_message(user_id, message.text)
        while response.upper() == response:
            response = process_message(user_id, message.text)

        bot.reply_to(message, response, reply_markup=create_main_keyboard())
        # Сохранение ответа бота в базу данных
        save_message(user_id, f"{response}", num_db_msgs)


num_db_msgs = 3
max_msg_len = 40


def process_message(user_id, text):
    global max_msg_len, num_db_msgs
    # Получение всей истории диалога из базы данных
    msgs = get_message_history(user_id, num_db_msgs)
    dialog_history = "\n".join(msgs) + "\n"

    # Кодирование и генерация ответа
    in_prompt = f"{dialog_history}"
    inpt = tok.encode(in_prompt, return_tensors="pt")
    max_len = max_msg_len + len(in_prompt)
    out = model.generate(inpt.to(device), max_length=max_len, repetition_penalty=30.0,
                         do_sample=True, top_k=5, top_p=0.75, temperature=1)
    response = strip_me(tok.decode(out[0]), len(msgs))

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
    # Получаем последние n сообщений из базы данных
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


bot.infinity_polling()

# Закрытие соединения с базой данных при завершении работы бота
atexit.register(lambda: conn.close())
