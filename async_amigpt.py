import atexit
import asyncio
import random
import logging
import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pymorphy3
import regex as re
import torch
from aiogram import Bot, Dispatcher, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message
from aiogram.exceptions import TelegramAPIError
from aiogram.client.bot import Bot
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import MBartForConditionalGeneration, AutoTokenizer

import tokens_amigpt

# Import aiosqlite for asynchronous SQLite operations
import aiosqlite

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(77)
torch.manual_seed(77)

# Initialize the bot with your token
bot_token = tokens_amigpt.BOT_TOKEN
bot_username = tokens_amigpt.BOT_NAME
bot = Bot(token=bot_token)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Load the GPT-2 model and tokenizer
model_name = "models/amigpt5"
tok = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Update device initialization to support Macs with MPS
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Device: {device.type}")
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
p = 0.5  # Probability for random summary messages in group chats

# Create a ThreadPoolExecutor for running model inferences
model_executor = ThreadPoolExecutor(max_workers=5)  # Adjust the number of workers as needed


# Function to initialize the database
async def init_db():
    async with aiosqlite.connect('message_history.db') as db:
        await db.execute('PRAGMA journal_mode=WAL')  # Enable Write-Ahead Logging for better concurrency
        await db.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            user_id INTEGER,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        await db.execute('''
        CREATE INDEX IF NOT EXISTS idx_chat_id ON history (chat_id)
        ''')
        await db.execute('''
        CREATE TABLE IF NOT EXISTS chat_counters (
            chat_id INTEGER PRIMARY KEY,
            current_message_count INTEGER DEFAULT 0
        )
        ''')
        await db.execute(f'''
        CREATE TABLE IF NOT EXISTS chat_limits (
            chat_id INTEGER PRIMARY KEY,
            max_messages INTEGER DEFAULT {max_hist_default}
        )
        ''')
        await db.commit()
        logger.info("Database initialized successfully")


# Function to sanitize user inputs
def sanitize_input(text):
    sanitized_text = re.sub(r'[^\w\s,.!?@-]', '', text)
    return sanitized_text.strip()


# Function to summarize text using the mBART model
def summarize(text, max_length=500):
    try:
        text1 = text
        input_ids = tokenizer_summ(
            [text1],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(device)

        output_ids = model_summ.generate(
            input_ids=input_ids,
            # max_length=max_length,
            num_beams=3,
            no_repeat_ngram_size=3,
        )[0]

        summary = tokenizer_summ.decode(output_ids, skip_special_tokens=True)
        return summary
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return ""


async def send_message(chat_id, text, **kwargs):
    await bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML', **kwargs)


async def reply_message(message: Message, text, **kwargs):
    await message.reply(text, parse_mode='HTML', **kwargs)


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
        logger.error(f"strip_me error: {e}")
        return "что?"


# Function to split long messages
def split_message(message):
    return [message[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(message), MAX_MESSAGE_LENGTH)]


# Function to create the main keyboard
def create_main_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text='Help')
    builder.button(text='Reset')
    builder.adjust(2)  # Arranges buttons into rows with 2 buttons per row
    return builder.as_markup(resize_keyboard=True)


# Create a router
from aiogram import Router

router = Router()


# Handler for /start and /help commands
@router.message(F.text.in_({'/start', '/help'}))
async def send_welcome(message: Message):
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
    await reply_message(message, help_message, reply_markup=create_main_keyboard())


# Handler for /reset command
@router.message(F.text == '/reset')
async def reset_history(message: Message):
    chat_id = message.chat.id
    try:
        async with aiosqlite.connect('message_history.db') as db:
            await db.execute('''
                INSERT OR REPLACE INTO chat_counters (chat_id, current_message_count)
                VALUES (?, 0)
            ''', (chat_id,))
            await db.commit()
        await reply_message(message,
                            "Your conversation history has been reset. The bot forgot all your previous responses.",
                            reply_markup=create_main_keyboard()
                            )
        logger.info(f"Chat history reset for chat_id: {chat_id}")
    except Exception as e:
        logger.error(f"Error resetting history for chat_id {chat_id}: {e}")
        await reply_message(message, "An error occurred while resetting your history.")


# Function to increment the message count
async def increment_message_count(chat_id):
    try:
        async with aiosqlite.connect('message_history.db') as db:
            async with db.execute('SELECT current_message_count FROM chat_counters WHERE chat_id = ?',
                                  (chat_id,)) as cursor:
                row = await cursor.fetchone()
                current_count = row[0] if row else 0

            async with db.execute('SELECT max_messages FROM chat_limits WHERE chat_id = ?', (chat_id,)) as cursor:
                row = await cursor.fetchone()
                max_count = row[0] if row else max_hist_default

            if current_count < max_count:
                current_count += 1
                await db.execute(
                    'INSERT OR REPLACE INTO chat_counters (chat_id, current_message_count) VALUES (?, ?)',
                    (chat_id, current_count)
                )
                await db.commit()
    except Exception as e:
        logger.error(f"Error incrementing message count for chat_id {chat_id}: {e}")


# Handler for all messages
@router.message()
async def handle_message(message: Message):
    chat_id = message.chat.id
    user_id = message.from_user.id

    try:
        user_text = sanitize_input(message.text)

        if message.chat.type in ['group', 'supergroup']:
            # Save the message to history
            await save_message(chat_id, user_id, user_text)

            # Increment message count
            await increment_message_count(chat_id)

            if 'summarize' in user_text.lower():
                # Extract number of messages to summarize
                match = re.search(r'summarize\s*(\d+)?', user_text.lower())
                if match:
                    n = match.group(1)
                    if n:
                        n = int(n)
                        if n <= 0:
                            await message.reply("Please provide a positive number of messages to summarize.")
                            return
                        n = min(300, n)
                    else:
                        n = 100  # Default to 100 messages
                else:
                    n = 100

                # Get the last n messages from the group
                msgs = await get_message_history(chat_id, n)
                if msgs:
                    summarized_text = summarize("<s>" + "\n<s>".join(msgs) + "\n", max_length=3000)
                    if summarized_text:
                        await reply_message(message, f"Summary of the last {n} messages:\n\n{summarized_text}")
                else:
                    await reply_message(message, "Not enough messages to summarize.")
                return

            # Check if bot is mentioned or in insane mode
            if is_insane.get(chat_id,
                             False) or f"@{bot_username}" in user_text or message.reply_to_message and message.reply_to_message.from_user.id == bot.id:
                # Remove bot mention if present
                text = user_text.replace(f"@{bot_username}", '').strip()
                max_summarize_length = 1500

                response = await process_message(chat_id, text, max_summarize_length)
                while response.upper() == response:
                    response = await process_message(chat_id, text, max_summarize_length)

                await reply_message(message, response)
                await save_message(chat_id, None, response)
                await increment_message_count(chat_id)
            else:
                # With probability p, the bot sends a summary message
                if random.random() < p:
                    msgs = await get_message_history(chat_id, 100)
                    if msgs:
                        summarized_text = summarize("\n".join(msgs), max_length=3000)
                        if summarized_text:
                            # Generate a response based on the summary
                            in_prompt = f"<s>{summarized_text}\n"
                            inpt = tok.encode(in_prompt, return_tensors="pt").to(device)
                            max_len = max_response_size + len(in_prompt)

                            # Run model inference asynchronously
                            loop = asyncio.get_event_loop()
                            out = await loop.run_in_executor(
                                model_executor,  # Use model_executor here
                                functools.partial(
                                    model.generate,
                                    inpt,
                                    max_length=max_len,
                                    repetition_penalty=5.0,
                                    do_sample=True,
                                    top_k=5,
                                    top_p=0.95,
                                    temperature=0.7,
                                    no_repeat_ngram_size=3
                                )
                            )

                            response = strip_me(tok.decode(out[0]), 1, min_check_length)
                            await send_message(chat_id, response)
                            await save_message(chat_id, None, response)
        else:
            if user_text.lower() == 'help':
                await send_welcome(message)
                return
            elif user_text.lower() == 'reset':
                await reset_history(message)
                return
            elif "get hist" in user_text.lower():
                await print_user_hist(message, chat_id)
                return
            elif tokens_amigpt.INSANITY_ON in user_text:
                is_insane[chat_id] = True
                await reply_message(message, "I'm INSANE!!!", reply_markup=create_main_keyboard())
                return
            elif tokens_amigpt.INSANITY_OFF in user_text:
                is_insane[chat_id] = False
                await reply_message(message, "I'm NOT INSANE.", reply_markup=create_main_keyboard())
                return
            elif "set hist size" in user_text.lower():
                match = re.search(r'\d+', user_text)
                if match:
                    max_history = int(match.group())
                    try:
                        async with aiosqlite.connect('message_history.db') as db:
                            await db.execute(
                                'INSERT OR REPLACE INTO chat_limits (chat_id, max_messages) VALUES (?, ?)',
                                (chat_id, max_history)
                            )
                            await db.commit()
                        await reply_message(message, f"Max history size set to {max_history}",
                                            reply_markup=create_main_keyboard())
                        logger.info(f"Set history size to {max_history} for chat_id {chat_id}")
                    except Exception as e:
                        logger.error(f"Error setting history size for chat_id {chat_id}: {e}")
                        await reply_message(message,
                                            "An error occurred while setting history size.",
                                            reply_markup=create_main_keyboard()
                                            )
                else:
                    await reply_message(message,
                                        "Please provide a valid number for history size.",
                                        reply_markup=create_main_keyboard()
                                        )
                return
            elif user_text.lower().startswith('summarize'):
                # Extract number of messages to summarize
                match = re.search(r'summarize\s*(\d+)?', user_text.lower())
                message_count = await get_user_hist_size(chat_id)
                if match:
                    n = match.group(1)
                    if n:
                        n = int(n)
                        if n <= 0:
                            await message.reply(
                                "Please provide a positive number of messages to summarize.",
                                reply_markup=create_main_keyboard()
                            )
                            return
                        else:
                            n = min(n, 100)  # Limit to 300 messages max
                    else:
                        n = message_count
                else:
                    n = message_count

                # Get the user's message history
                msgs = await get_message_history(chat_id, n)
                if msgs:
                    to_summ = "<s>" + "\n<s>".join(msgs) + "\n"
                    summarized_text = summarize(to_summ, max_length=1000)
                    if summarized_text:
                        await message.reply(
                            f"Summary of your last {len(msgs)} messages:\n\n{summarized_text}",
                            reply_markup=create_main_keyboard()
                        )
                    else:
                        await message.reply(
                            "An error occurred while summarizing your messages.",
                            reply_markup=create_main_keyboard()
                        )
                else:
                    await message.reply(
                        "Not enough messages to summarize.",
                        reply_markup=create_main_keyboard()
                    )
                return

            # Save the message to history
            await save_message(chat_id, user_id, user_text)

            # Increment message count
            await increment_message_count(chat_id)

            max_summarize_length = 300

            response = await process_message(chat_id, user_text, max_summarize_length)
            while response.upper() == response:
                response = await process_message(chat_id, user_text, max_summarize_length)

            await reply_message(message, response, reply_markup=create_main_keyboard())
            await save_message(chat_id, None, response)
            await increment_message_count(chat_id)
    except TelegramAPIError as e:
        logger.error(f"Telegram API error: {e}")
    except Exception as e:
        logger.error(f"Error handling message from chat_id {chat_id}: {e}")
        await reply_message(message, "An error occurred while processing your message.")


async def get_user_hist_size(chat_id):
    try:
        async with aiosqlite.connect('message_history.db') as db:
            async with db.execute(
                    'SELECT current_message_count FROM chat_counters WHERE chat_id = ?',
                    (chat_id,)
            ) as cursor:
                row = await cursor.fetchone()
                message_count = row[0] if row else 0

        return message_count
    except Exception as e:
        logger.error(f"Error while getting history size for chat_id {chat_id}: {e}")
        return 0


# Function to process the user's message and generate a response
async def process_message(chat_id, text, max_summarize_length=300):
    global max_response_size, min_check_length
    try:
        message_count = await get_user_hist_size(chat_id)

        msgs = await get_message_history(chat_id, message_count)
        dialog_history = "<s>" + "\n<s>".join(msgs) + "\n"
        in_prompt = f"{dialog_history}"

        if do_summarize and len(in_prompt) > max_summarize_length:
            summary_text = summarize(in_prompt, max_length=max_summarize_length)
            if summary_text:
                in_prompt = "<s>" + text + "\n<s>" + summary_text

            current_newline_count = in_prompt.count("\n")
            if current_newline_count < len(msgs):
                in_prompt += "\n" * (len(msgs) - current_newline_count)

        # Prepare input for the model
        # logger.error(f"In prompt: {in_prompt + '\n'}")
        inpt = tok.encode(in_prompt + '\n', return_tensors="pt").to(device)
        max_len = max_response_size + len(in_prompt)

        # Run model inference asynchronously
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            model_executor,  # Use model_executor here
            functools.partial(
                model.generate,
                inpt,
                max_length=max_len,
                repetition_penalty=5.0,
                do_sample=True,
                top_k=5,
                top_p=0.95,
                temperature=0.7,
                no_repeat_ngram_size=3
            )
        )

        # Process the model output
        response = strip_me(tok.decode(out[0]), len(msgs), min_check_length)
        return response
    except Exception as e:
        logger.error(f"Process message error for chat_id {chat_id}: {e}")
        return "I couldn't process that message."


async def print_user_hist(message, chat_id):
    async with aiosqlite.connect('message_history.db') as db:
        async with db.execute('SELECT current_message_count FROM chat_counters WHERE chat_id = ?',
                              (chat_id,)) as cursor:
            row = await cursor.fetchone()
            message_count = row[0] if row else 0

    msgs = await get_message_history(chat_id, message_count)
    dialog_history = "\n\n".join(msgs) + "\n"

    await reply_message(message, dialog_history, reply_markup=create_main_keyboard())


# Function to save a message to the database
async def save_message(chat_id, user_id, message):
    try:
        async with aiosqlite.connect('message_history.db') as db:
            await db.execute(
                'INSERT INTO history (chat_id, user_id, message) VALUES (?, ?, ?)',
                (chat_id, user_id, message)
            )
            await db.commit()
    except Exception as e:
        logger.error(f"Save message error for chat_id {chat_id}: {e}")


# Function to retrieve message history
async def get_message_history(chat_id, n=3):
    try:
        async with aiosqlite.connect('message_history.db') as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                    'SELECT message FROM history WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?',
                    (chat_id, n)
            ) as cursor:
                rows = await cursor.fetchall()
                return [row['message'] for row in reversed(rows)]
    except Exception as e:
        logger.error(f"Get message history error for chat_id {chat_id}: {e}")
        return []


async def get_user_message_history(chat_id, user_id, n=300):
    try:
        async with aiosqlite.connect('message_history.db') as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                    'SELECT message FROM history WHERE chat_id = ? AND user_id = ? ORDER BY timestamp DESC LIMIT ?',
                    (chat_id, user_id, n)
            ) as cursor:
                rows = await cursor.fetchall()
                return [row['message'] for row in reversed(rows)]
    except Exception as e:
        logger.error(f"Get user message history error for chat_id {chat_id}, user_id {user_id}: {e}")
        return []


# Function to clear chat history
async def clear_chat_history(chat_id):
    try:
        async with aiosqlite.connect('message_history.db') as db:
            await db.execute('DELETE FROM history WHERE chat_id = ?', (chat_id,))
            await db.execute('DELETE FROM chat_counters WHERE chat_id = ?', (chat_id,))
            await db.execute('DELETE FROM chat_limits WHERE chat_id = ?', (chat_id,))
            await db.commit()
            logger.info(f"Cleared chat history for chat_id {chat_id}")
    except Exception as e:
        logger.error(f"Clear chat history error for chat_id {chat_id}: {e}")


# Ensure model_executor is shut down on exit
@atexit.register
def shutdown_executor():
    model_executor.shutdown()


# Include the router in the dispatcher
dp.include_router(router)


# Start the bot
async def main():
    await init_db()
    await dp.start_polling(bot, skip_updates=True)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")
