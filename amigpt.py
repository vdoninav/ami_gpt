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

import tokens_amigpt  # Replace with your actual tokens module

# Set random seeds for reproducibility
np.random.seed(77)
torch.manual_seed(77)

# Initialize the bot with your token
bot_token = tokens_amigpt.BOT_TOKEN
bot_username = tokens_amigpt.BOT_NAME
bot = telebot.TeleBot(bot_token)

# Load the GPT-2 model and tokenizer
model_name = "models/amigpt5"
tok = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Update device initialization to support Macs with MPS
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
stderr.write(f"Device: {device.type}\n")
model.to(device)

# Load the summarization model
do_summarize = True
if do_summarize:
    model_name_summ = "models/mbart_ruDialogSum"
    tokenizer_summ = AutoTokenizer.from_pretrained(model_name_summ)
    model_summ = MBartForConditionalGeneration.from_pretrained(model_name_summ)
    model_summ.to(device)
    model_summ.eval()

# Initialize morphological analyzer
morph = pymorphy3.MorphAnalyzer()

# Use a dictionary to store 'is_insane' status per chat
is_insane = {}

# Constants
MAX_MESSAGE_LENGTH = 4096
max_response_size = 40
min_check_length = 11
max_hist_default = 4

# Connect to the SQLite database and create tables for storing message history
conn = sqlite3.connect('message_history.db', check_same_thread=False)
cursor = conn.cursor()

# Create the history table with chat_id
cursor.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    user_id INTEGER,
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

cursor.execute('''
CREATE INDEX IF NOT EXISTS idx_chat_id ON history (chat_id)
''')
conn.commit()

# Create counters for users and chats
cursor.execute('''
CREATE TABLE IF NOT EXISTS user_counters (
    user_id INTEGER PRIMARY KEY,
    current_message_count INTEGER DEFAULT 0
)
''')
conn.commit()

cursor.execute('''
CREATE TABLE IF NOT EXISTS chat_counters (
    chat_id INTEGER PRIMARY KEY,
    current_message_count INTEGER DEFAULT 0
)
''')
conn.commit()

# Create tables for storing maximum message limits
cursor.execute(f'''
CREATE TABLE IF NOT EXISTS user_limits (
    user_id INTEGER PRIMARY KEY,
    max_messages INTEGER DEFAULT {max_hist_default}
)
''')
conn.commit()

cursor.execute(f'''
CREATE TABLE IF NOT EXISTS chat_limits (
    chat_id INTEGER PRIMARY KEY,
    max_messages INTEGER DEFAULT {max_hist_default}
)
''')
conn.commit()


# Function to summarize text using the mBART model
def summarize(text):
    try:
        input_ids = tokenizer_summ(
            [text],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(device)

        output_ids = model_summ.generate(
            input_ids=input_ids,
            num_beams=3,
            no_repeat_ngram_size=3,
        )[0]

        summary = tokenizer_summ.decode(output_ids, skip_special_tokens=True)
        return summary
    except Exception as e:
        stderr.write(f"Summarization error: {e}\n")
        return ""


# Function to check if a word is Russian
def is_russian_word(word, min_length=3):
    if len(word) <= min_length or word.isdigit():
        return True
    parts = word.split('-')
    return all(any(parse.is_known for parse in morph.parse(part)) for part in parts)


# Function to process the model's output and clean it up
def strip_me(input_string, n=1, min_length=3):
    try:
        pos = -1
        for _ in range(n):
            pos = input_string.find('\n', pos + 1)
            if pos == -1:
                return "DEBUG MESSAGE - не найдена n-я новая строка"

        sub_string = input_string[pos + 1:]

        cyr_match = re.search(r'\p{IsCyrillic}', sub_string)
        if not cyr_match:
            return "что?"

        sub_string = sub_string[cyr_match.start():]

        n = 300
        if len(sub_string) > n:
            punc_positions = [sub_string.find(punc) for punc in '.!?\n' if
                              sub_string.find(punc) != -1 and sub_string.find(punc) < n]
            if punc_positions:
                punc_positions = sorted(set(punc_positions))
                punc_pos = punc_positions[1] if len(punc_positions) > 1 else punc_positions[0]
                sub_string = sub_string[:punc_pos + 1].strip()
            else:
                sub_string = sub_string[:n].strip()
        else:
            punc_positions = [sub_string.find(punc) for punc in '.!?' if sub_string.find(punc) != -1]
            if punc_positions:
                punc_pos = min(punc_positions)
                sub_string = sub_string[:punc_pos + 1].strip()

        words = re.split(r'\s+', sub_string)
        filtered_words = [word for word in words if is_russian_word(word, min_length)]
        sub_string = ' '.join(filtered_words)

        if not re.search(r'[А-Яа-яЁё]', sub_string):
            return "что?"

        return sub_string
    except Exception as e:
        stderr.write(f"strip_me error: {e}\n")
        return "что?"


# Function to split long messages
def split_message(message):
    return [message[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(message), MAX_MESSAGE_LENGTH)]


# Function to create the main keyboard
def create_main_keyboard():
    keyboard = telebot.types.ReplyKeyboardMarkup(
        keyboard=[
            [telebot.types.KeyboardButton('Help'), telebot.types.KeyboardButton('Reset')]
        ],
        resize_keyboard=True
    )
    return keyboard


# Handler for /start and /help commands
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    help_message = (
        "Welcome to AmiGPT!\n\n"
        "Here are the commands you can use:\n"
        "- /start or /help: Show this help message and available commands.\n"
        "- /reset: Reset your conversation history. The bot will forget everything you have written before.\n"
        "- set hist size [number]: Set the number of messages to use from your conversation history for generating responses.\n"
        "- summarize [n]: Summarize the last n messages in the group (default is 100 if n is not provided).\n\n"
        "You can also activate 'insane' mode by sending a special keyword, or switch back to normal mode using another keyword.\n\n"
        "Have fun chatting!"
    )
    bot.reply_to(message, help_message, reply_markup=create_main_keyboard())


# Handler for /reset command
@bot.message_handler(commands=['reset'])
def reset_history(message):
    chat_id = message.chat.id
    cursor.execute('''
        INSERT OR REPLACE INTO chat_counters (chat_id, current_message_count)
        VALUES (?, 0)
    ''', (chat_id,))
    conn.commit()
    bot.reply_to(message, "Your conversation history has been reset. "
                          "The bot forgot all your previous responses.",
                 reply_markup=create_main_keyboard())


# Function to increment the message count
def increment_message_count(chat_id):
    cursor.execute('SELECT current_message_count FROM chat_counters WHERE chat_id = ?', (chat_id,))
    current_count = cursor.fetchone()

    if current_count:
        current_count = current_count[0]
        cursor.execute('SELECT max_messages FROM chat_limits WHERE chat_id = ?', (chat_id,))
        max_count = cursor.fetchone()
        max_count = max_count[0] if max_count else max_hist_default

        if current_count < max_count:
            current_count += 1
            cursor.execute('UPDATE chat_counters SET current_message_count = ? WHERE chat_id = ?',
                           (current_count, chat_id))
            conn.commit()
    else:
        cursor.execute('INSERT INTO chat_counters (chat_id, current_message_count) VALUES (?, ?)', (chat_id, 1))
        conn.commit()


# Handler for all messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Initialize 'is_insane' status for the chat if not present
    if chat_id not in is_insane:
        is_insane[chat_id] = False

    if message.chat.type in ['group', 'supergroup']:
        # Save the message to history
        save_message(chat_id, user_id, message.text)

        # Increment message count
        increment_message_count(chat_id)

        if 'summarize' in message.text.lower():
            # Extract number of messages to summarize
            match = re.search(r'summarize\s*(\d+)?', message.text.lower())
            if match:
                n = match.group(1)
                if n:
                    n = int(n)
                    if n <= 0:
                        bot.reply_to(message, "Please provide a positive number of messages to summarize.")
                        return
                else:
                    n = 100  # Default to 100 messages
            else:
                n = 100

            # Get the last n messages from the group
            msgs = get_message_history(chat_id, n)
            if msgs:
                summarized_text = summarize("\n".join(msgs))
                if summarized_text:
                    bot.reply_to(message, f"Summary of the last {n} messages:\n\n{summarized_text}")
            else:
                bot.reply_to(message, "Not enough messages to summarize.")
            return

        if is_insane.get(chat_id, False):
            response = process_message(chat_id, user_id, message.text)
            bot.reply_to(message, response, reply_markup=create_main_keyboard())

        if f"@{bot_username}" in message.text:
            text = message.text.replace(f"@{bot_username}", '').strip()
            response = process_message(chat_id, user_id, text)
            bot.reply_to(message, response, reply_markup=create_main_keyboard())
    else:
        if message.text.lower() == 'help':
            send_welcome(message)
            return
        elif message.text.lower() == 'reset':
            reset_history(message)
            return
        elif tokens_amigpt.INSANITY_ON in message.text:
            is_insane[chat_id] = True
            bot.reply_to(message, "I'm INSANE!!!", reply_markup=create_main_keyboard())
            return
        elif tokens_amigpt.INSANITY_OFF in message.text:
            is_insane[chat_id] = False
            bot.reply_to(message, "I'm NOT INSANE.", reply_markup=create_main_keyboard())
            return
        elif "set hist size" in message.text.lower():
            match = re.search(r'\d+', message.text)
            if match:
                max_history = int(match.group())
                cursor.execute('INSERT OR REPLACE INTO chat_limits (chat_id, max_messages) VALUES (?, ?)',
                               (chat_id, max_history))
                conn.commit()
                bot.reply_to(message, f"Max history size set to {max_history}", reply_markup=create_main_keyboard())
            else:
                bot.reply_to(message, "Please provide a valid number for history size.",
                             reply_markup=create_main_keyboard())
            return

        # Save the message to history
        save_message(chat_id, user_id, message.text)

        # Increment message count
        increment_message_count(chat_id)

        response = process_message(chat_id, user_id, message.text)
        while response.upper() == response:
            response = process_message(chat_id, user_id, message.text)

        bot.reply_to(message, response, reply_markup=create_main_keyboard())
        # Save the bot's response to the database
        save_message(chat_id, None, response)

        # Increment message count after the bot's response
        increment_message_count(chat_id)


# Function to process the user's message and generate a response
def process_message(chat_id, user_id, text):
    global max_response_size, min_check_length
    cursor.execute('SELECT current_message_count FROM chat_counters WHERE chat_id = ?', (chat_id,))
    count_row = cursor.fetchone()
    message_count = count_row[0] if count_row else 0

    msgs = get_message_history(chat_id, message_count)
    dialog_history = "<s>" + "\n<s>".join(msgs) + "\n"
    in_prompt = f"{dialog_history}"

    if do_summarize and len(in_prompt) > 300:
        summary_text = summarize(in_prompt)
        if summary_text:
            in_prompt = "<s>" + text + "\n<s>" + summary_text

        current_newline_count = in_prompt.count("\n")
        if current_newline_count < len(msgs):
            in_prompt += "\n" * (len(msgs) - current_newline_count)

    inpt = tok.encode(in_prompt + '\n', return_tensors="pt").to(device)
    max_len = max_response_size + len(in_prompt)
    out = model.generate(inpt, max_length=max_len, repetition_penalty=5.0,
                         do_sample=True, top_k=5, top_p=0.95, temperature=0.7, no_repeat_ngram_size=3)
    response = strip_me(tok.decode(out[0]), len(msgs), min_check_length)

    return response


# Function to save a message to the database
def save_message(chat_id, user_id, message):
    cursor.execute('INSERT INTO history (chat_id, user_id, message) VALUES (?, ?, ?)', (chat_id, user_id, message))
    conn.commit()


# Function to retrieve message history
def get_message_history(chat_id, n=3):
    cursor.execute('SELECT message FROM history WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?', (chat_id, n))
    rows = cursor.fetchall()
    return [row[0] for row in reversed(rows)]


# Function to clear chat history
def clear_chat_history(chat_id):
    cursor.execute('DELETE FROM history WHERE chat_id = ?', (chat_id,))
    conn.commit()


stderr.write("Initialized successfully\n")

bot.infinity_polling()

# Close the database connection when the bot stops
atexit.register(lambda: conn.close())
