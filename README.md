# AmiGPT Telegram Bot

AmiGPT is a Telegram bot that uses a pre-trained GPT-2 model to communicate with users in private chats and groups. The bot generates responses based on the current conversation context and stores interaction history in an SQLite database, allowing it to take into account previous messages when forming responses. The bot also supports managing the number of messages stored in the context and allows users to reset or adjust the message limit.

## Features

- **AI-powered conversation**: Utilizes the GPT-2 model to generate responses based on the conversation context.
- **Message history**: Saves and retrieves dialogue history from an SQLite database.
- **History management**: Users can reset the number of messages taken into account or adjust the message limit.
- **Modes**: Can operate in "normal" or "insane" mode, changing the nature of responses.
- **Command handling**: Supports `/start`, `/help`, `/reset`, and allows setting message limits.
- **Group chat support**: The bot processes mentions in group chats and responds to messages.
- **Russian word filtering**: The bot filters messages by checking if they contain valid Russian words (with support for hyphenated words) and removes messages that don’t contain Cyrillic characters.

## Commands

- **`/start`**: Starts the bot and displays a welcome message.
- **`/help`**: Provides help and usage instructions for the bot.
- **`/reset`**: Resets the message counter used for generating responses back to zero. After that, the bot will start counting messages again.
- **`set hist size [number]`**: Sets the maximum number of messages that will be used in the context for generating responses.

### Special Commands

- **Activate "insane" mode**: Send the trigger word defined in `INSANITY_ON` to activate the "insane" mode.
- **Deactivate "insane" mode**: Send the trigger word defined in `INSANITY_OFF` to return to normal mode.
- **Set the message limit for history**: Use the command `set hist size [number]` to specify how many messages the bot will keep in memory for each user.

### Group Interaction

- The bot responds to messages where its name is mentioned.
- In "insane" mode, the bot may respond to more messages or behave more unpredictably.

## Message History Logic

The bot saves message history in an SQLite database. Instead of deleting messages upon reset, the bot simply starts ignoring older messages until users send new ones. The message limit can be configured for each user using the `set hist size` command.

With each new message or response from the neural network, the bot increases the number of messages it takes into account in the context, up to the predefined limit. The limit can be reset at any time using `/reset`.

### Database Structure

1. **`history` table**:
   - Stores all user messages with timestamps.
   - Columns: `id`, `user_id`, `message`, `timestamp`.

2. **`user_counters` table**:
   - Tracks the current number of messages that the bot uses for responses.
   - Columns: `user_id`, `current_message_count`.

3. **`user_limits` table**:
   - Stores the maximum number of messages the bot will consider for each user.
   - Columns: `user_id`, `max_messages`.

## Technical Details

- **Model**: GPT-2 with pre-trained weights loaded via the HuggingFace `transformers` library.
- **Platform**: Python, using PyTorch for model execution.
- **Text processing**: The bot filters messages based on the presence of valid Russian words and removes sequences of numbers or non-Cyrillic characters.
- **Responses**: Response length is limited to 40 characters by default (configurable in the code).

## Customization

- **Message history**: Adjust the number of stored messages with the variable `max_hist_default` or by using the command `set hist size`.
- **Text filtering**: The bot checks words for the presence of Cyrillic characters and filters out invalid words. You can customize the minimum word length for validation in the `is_russian_word` function.

## Conclusion

AmiGPT is a flexible Telegram bot that can be used for intelligent conversations with users, as well as customized for various interaction scenarios. It allows full control over the message history and response generation logic.

##
`Built with Python 3.12`

`Compatible with Python 3.9 or newer`

