# AmiGPT Telegram Bot

AmiGPT — это Telegram-бот, использующий обученную модель GPT-2 для общения с пользователями в приватных чатах и группах. Бот генерирует ответы на основе текущего контекста разговора и сохраняет историю взаимодействий в базе данных SQLite, что позволяет ему учитывать предыдущие сообщения при формировании ответов.

## Возможности

- **Искусственный интеллект для общения**: Использует модель GPT-2 для генерации ответов с учётом контекста.
- **История сообщений**: Сохраняет и извлекает историю диалогов из базы данных SQLite.
- **Режимы работы**: Может работать в "нормальном" и "безумном" режимах, меняя характер ответов.
- **Обработка команд**: Поддерживает команды `/start`, `/help` и `/reset`.
- **Поддержка групповых чатов**: Обрабатывает упоминания и взаимодействия в группах.

## Команды

- **`/start`**: Запускает бота и отображает приветственное сообщение.
- **`/help`**: Предоставляет справку о работе с ботом.
- **`/reset`**: Очищает историю сообщений пользователя.

### Специальные команды в приватном чате

- **Активация "безумного" режима**: Отправьте триггерное слово, определенное в `INSANITY_ON`, чтобы включить "безумный" режим.
- **Деактивация "безумного" режима**: Отправьте триггерное слово, определенное в `INSANITY_OFF`, чтобы вернуться в нормальный режим.
- **Установка лимита на количество сообщений в базе данных**: Используйте команду `set db size [число]`, чтобы задать количество сообщений, сохраняемых в истории для каждого пользователя.

### Взаимодействие в групповых чатах

- Бот отвечает, когда его имя упоминается в сообщении.
- В "безумном" режиме бот может реагировать на большее количество сообщений или вести себя менее предсказуемо.

## База данных

Бот использует базу данных SQLite для хранения истории сообщений. База данных создается автоматически при запуске бота, создается таблица для хранения истории сообщений и индекс для ускорения поиска.

## Кастомизация

- **Обработка сообщений**: Вы можете настроить, как бот обрабатывает и генерирует ответы, изменив функцию `process_message`.
- **Управление историей**: Отрегулируйте количество сохраняемых сообщений, изменив значения переменных `num_db_msgs` и `max_msg_len`.
