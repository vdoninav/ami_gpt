# AmiGPT Telegram Bot

AmiGPT is a Telegram bot powered by a pre-trained GPT-2 model that communicates with users in private chats and groups. It generates responses based on the conversation context, utilizing an SQLite database to store and retrieve message history. The bot also supports message summarization, custom message limits, and can operate in "normal" or "insane" mode.

## Features

- **AI-Powered Conversation**: Generates responses using the GPT-2 model, taking into account conversation context.
- **Message History Management**: Saves and retrieves dialogue history using an SQLite database.
- **Message Summarization**:
  - **Summarize Command**: Users can request a summary of their own messages (up to 300) using the `summarize` command in private chats.
  - **Automatic Summarization**: Summarizes long conversation histories using an mBART model to keep the context concise.
- **Adjustable Message Limits**: Users can set the number of past messages used for context or reset the message counter.
- **Modes**: Supports "normal" and "insane" modes, changing the nature of responses on a per-chat basis.
- **Group Chat Support**:
  - **Mentions**: Responds to messages where the bot is mentioned.
  - **Replies**: Can respond to replies to its own messages in group chats.
- **Russian Language Filtering**: Filters messages to ensure they contain valid Russian words, supporting hyphenated words and ignoring messages without Cyrillic characters.
- **Device Compatibility**: Automatically selects the best available device (CUDA, MPS, or CPU) for running the models, supporting Macs with MPS.

## Commands

- **`/start`**: Starts the bot and displays a welcome message.
- **`/help`**: Provides help and usage instructions for the bot.
- **`/reset`**: Resets the message counter used for generating responses, starting with zero messages for the next response.
- **`set hist size [number]`**: Sets the maximum number of messages the bot will consider when generating responses.
- **`summarize [n]`**: Summarizes your last `n` messages (up to 300) in private chats. If `n` is not provided, summarizes the last 300 messages.

### Special Commands

- **Activate "Insane" Mode**: Send the word defined in `INSANITY_ON` to switch the bot to an unpredictable, "insane" mode. This mode is managed per chat.
- **Deactivate "Insane" Mode**: Send the word defined in `INSANITY_OFF` to return the bot to normal mode.
- **Set History Size**: Use `set hist size [number]` to specify how many past messages the bot will use in the context for responses.

### Group Interaction

- **Mentions**: The bot responds to mentions in group chats.
- **Replies**: The bot responds when users reply to its messages in group chats.
- **"Insane" Mode**: In "insane" mode, the bot may respond to more messages or behave in a less predictable manner. This mode is now managed per chat.

## Message Summarization

### Summarize Command

Users in private chats can request a summary of their own messages:

- **Usage**: Send `summarize` or `summarize n` in a private chat.
  - If `n` is provided, the bot summarizes your last `n` messages (up to 300).
  - If `n` is not provided, the bot summarizes your last 300 messages.

### Automatic Summarization

If the conversation history exceeds a certain length, the bot automatically summarizes it using the mBART model. This helps keep the context concise for generating accurate responses.

## Message History Logic

AmiGPT saves all messages in an SQLite database. When a user resets their message history with `/reset`, the bot does not delete messages but starts using fewer messages in context (starting with zero). As the user continues chatting, the bot increases the number of messages used for context until it reaches the maximum limit, which can be adjusted.

The message limit can be customized for each user or chat with the `set hist size` command.

### Database Structure

1. **`history` table**:
   - Stores all messages with timestamps.
   - Columns: `id`, `chat_id`, `user_id`, `message`, `timestamp`.

2. **`user_counters` table**:
   - Tracks how many past messages the bot uses for generating responses in private chats.
   - Columns: `user_id`, `current_message_count`.

3. **`chat_counters` table**:
   - Tracks how many past messages the bot uses for generating responses in group chats.
   - Columns: `chat_id`, `current_message_count`.

4. **`user_limits` table**:
   - Stores the maximum number of messages to be used in responses for each user in private chats.
   - Columns: `user_id`, `max_messages`.

5. **`chat_limits` table**:
   - Stores the maximum number of messages to be used in responses for each group chat.
   - Columns: `chat_id`, `max_messages`.

## Technical Details

- **Models**:
  - **GPT-2**: Used for conversation generation (loaded via HuggingFace `transformers` library).
  - **mBART**: Used for summarizing long conversation histories.
- **Platform**: Python 3.12, using PyTorch for running models.
- **Device Compatibility**:
  - Automatically runs on the best available device:
    - **CUDA**: For systems with NVIDIA GPUs.
    - **MPS**: For Macs with Apple Silicon (M1/M2 chips).
    - **CPU**: If no GPU is available.
  - The device selection is handled in the code:
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    ```
- **Text Processing**:
  - Filters out messages that don't contain valid Russian words.
  - Supports hyphenated words and ignores messages without Cyrillic characters.
- **Message Summarization**:
  - Summarizes conversation history exceeding a certain length using the mBART model.
  - Supports summarization of up to 300 messages per request in private chats.
- **Response Generation**:
  - Responses are limited to a configurable size (default 40 tokens).
  - Supports parameters like `temperature`, `top_k`, and `top_p` for controlling the text generation.
- **Per-Chat Customization**:
  - **"Insane" Mode**: Managed per chat, allowing different chats to have different modes.
  - **Message Limits**: Can be set individually for users and group chats.

## Customization

- **History Size**:
  - Change the default number of past messages stored in the context with the `max_hist_default` variable.
  - Adjust per user or chat with the `set hist size` command.
- **Text Validation**:
  - Adjust the minimum word length for filtering Russian words in the `is_russian_word` function.
- **Summarization**:
  - Enable or disable the summarization feature by toggling the `do_summarize` flag in the bot's configuration.
- **Response Length**:
  - Control the maximum length of generated responses using the `max_response_size` variable.
- **"Insane" Mode**:
  - Customize the trigger words for activating or deactivating "insane" mode in the `amigpt_tokens` module.

## Example Setup

1. **Start the Bot**:
   - Use the `/start` or `/help` command to initialize the bot and get usage instructions.
2. **Reset Message Counter**:
   - Use `/reset` to reset how many messages the bot considers when generating a response.
3. **Set Message Limit**:
   - Use `set hist size [number]` to set the number of past messages the bot uses in context.
4. **Activate "Insane" Mode**:
   - Send the word defined in `INSANITY_ON` to activate "insane" mode in the current chat.
5. **Use the Summarize Command**:
   - In private chats, send `summarize` or `summarize n` to get a summary of your last `n` messages.

## Conclusion

AmiGPT is a highly configurable Telegram bot designed for intelligent conversation management. It supports Russian language filtering, message summarization, and can be customized for various interaction needs. With built-in history management, per-chat customization, and response generation logic, it offers flexible tools for engaging with users in private and group chats.

---

*Built with Python 3.12*

*Compatible with Python 3.9 or newer*
