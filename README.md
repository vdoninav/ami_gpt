# AmiGPT Telegram Bot

AmiGPT is a Telegram bot powered by a pre-trained GPT-2 model that communicates with users in private chats and groups. It generates responses based on the conversation context, utilizing an SQLite database to store and retrieve message history. The bot also supports message summarization, custom message limits, and can operate in "normal" or "insane" mode.

## Features

- **AI-Powered Conversation**: Generates responses using the GPT-2 model, taking into account conversation context.
- **Message History Management**: Saves and retrieves dialogue history using an SQLite database.
- **Message Summarization**: Summarizes long conversation histories using an mBART model to keep the context concise.
- **Adjustable Message Limits**: Users can set the number of past messages used for context or reset the message counter.
- **Modes**: Supports "normal" and "insane" modes, changing the nature of responses.
- **Group Chat Support**: Responds to messages where the bot is mentioned.
- **Russian Language Filtering**: Filters messages to ensure they contain valid Russian words, supporting hyphenated words and ignoring messages without Cyrillic characters.

## Commands

- **`/start`**: Starts the bot and displays a welcome message.
- **`/help`**: Provides help and usage instructions for the bot.
- **`/reset`**: Resets the message counter used for generating responses, starting with zero messages for the next response.
- **`set hist size [number]`**: Sets the maximum number of messages the bot will consider when generating responses.
- **"Insane" Mode**: Can be activated or deactivated with predefined trigger words set in the code.

### Special Commands

- **Activate "Insane" Mode**: Send the word defined in `INSANITY_ON` to switch the bot to an unpredictable, "insane" mode.
- **Deactivate "Insane" Mode**: Send the word defined in `INSANITY_OFF` to return the bot to normal mode.
- **Set History Size**: Use `set hist size [number]` to specify how many past messages the bot will use in the context for responses.

### Group Interaction

- The bot responds to mentions in group chats.
- In "insane" mode, the bot may respond to more messages or behave in a less predictable manner.

## Message History Logic

AmiGPT saves all user messages in an SQLite database. When a user resets their message history with `/reset`, the bot does not delete messages but starts using fewer messages in context (starting with zero). As the user continues chatting, the bot increases the number of messages used for context until it reaches the maximum limit, which can be adjusted.

The message limit can be customized for each user with the `set hist size` command.

### Database Structure

1. **`history` table**:
   - Stores all user messages with timestamps.
   - Columns: `id`, `user_id`, `message`, `timestamp`.

2. **`user_counters` table**:
   - Tracks how many past messages the bot uses for generating responses.
   - Columns: `user_id`, `current_message_count`.

3. **`user_limits` table**:
   - Stores the maximum number of messages to be used in responses for each user.
   - Columns: `user_id`, `max_messages`.

## Technical Details

- **Models**:
  - GPT-2 (loaded via HuggingFace `transformers` library) is used for conversation generation.
  - mBART is used for summarizing long conversation histories.
- **Platform**: Python 3.12, using PyTorch for running models.
- **Device Compatibility**: Automatically runs on CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU depending on the available hardware.
- **Text Processing**:
  - Filters out messages that don't contain valid Russian words.
  - Removes non-Cyrillic characters and sequences of digits from messages.
- **Message Summarization**: If the conversation history exceeds a certain length, it is summarized using the mBART model.
- **Response Generation**: Responses are limited to a configurable size (default 40 characters), with support for temperature, top-k, and top-p parameters for controlling generation.

## Customization

- **History Size**: Change the number of past messages stored in the context with the `max_hist_default` variable or the `set hist size` command.
- **Text Validation**: Adjust the minimum word length for filtering Russian words in the `is_russian_word` function.
- **Summarization**: Enable or disable the summarization feature by toggling the `do_summarize` flag in the bot's configuration.
- **Response Length**: Control the maximum length of generated responses using the `max_response_size` variable.

## Example Setup

1. **Start the bot**: The bot uses `/start` and `/help` commands to initialize and guide users.
2. **Reset Message Counter**: Use `/reset` to reset how many messages the bot considers when generating a response.
3. **Set Message Limit**: Use `set hist size [number]` to set the number of past messages the bot uses in context.

## Conclusion

AmiGPT is a highly configurable Telegram bot designed for intelligent conversation management. It supports Russian language filtering, message summarization, and can be customized for various interaction needs. With built-in history management and response generation logic, it offers flexible tools for engaging with users in private and group chats.

##
`Built with Python 3.12`

`Compatible with Python 3.9 or newer`
