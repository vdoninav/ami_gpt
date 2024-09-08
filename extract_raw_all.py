import os
import regex
import pymorphy3
from bs4 import BeautifulSoup

morph = pymorphy3.MorphAnalyzer()


def is_russian_word(word, min_length):
    # Пропускаем слова, которые меньше заданной длины или содержат только цифры
    if len(word) < min_length or word.isdigit():
        return True

    parses = morph.parse(word)
    # Если pymorphy3 не смог найти ни одной валидной формы, вероятно, слово не русское
    return any(parse.is_known for parse in parses)


def process_message(text, min_length=6):
    # Очищаем текст, оставляя только кириллицу, пробелы, дефисы, знаки препинания и цифры
    cleaned_text = regex.sub(r'[^\p{Cyrillic}\s.,?!-\d-]', '', text.get_text(strip=True))
    # Убираем последовательности из четырех и более цифр
    cleaned_text = regex.sub(r'\d{4,}', '', cleaned_text)
    # Убираем лишние пробелы
    cleaned_text = regex.sub(r'\s+', ' ', cleaned_text)
    # Убираем повторяющиеся знаки препинания
    cleaned_text = regex.sub(r'([ !,.?])\1+', ' ', cleaned_text)

    # Фильтруем слова, оставляя только русские, длиной больше min_length
    words = cleaned_text.split()
    filtered_words = []
    for word in words:
        # Если слово содержит дефис, разбиваем его на части и проверяем каждую часть
        if '-' in word:
            parts = word.split('-')
            if all(is_russian_word(part, min_length) for part in parts):
                filtered_words.append(word)
        else:
            if is_russian_word(word, min_length):
                filtered_words.append(word)

    return ' '.join(filtered_words)


def contains_cyrillic(input_string):
    return bool(regex.search(r'\p{IsCyrillic}', input_string))


directory = 'htmls'  # Путь к папке с экспортами Telegram
output_file = 'texts/output_all_8_with_nums.txt'

previous_message = ""  # Переменная для хранения предыдущего сообщения
previous_referenced_text = ""  # Переменная для хранения предыдущего текста на который был ответ

with open(output_file, 'w', encoding='utf-8') as out_file:
    for curr_dir in os.listdir(directory):
        full_path = os.path.join(directory, curr_dir)
        if os.path.isdir(full_path):
            for file in os.listdir(full_path):
                if file.endswith('.html'):
                    with open(os.path.join(full_path, file), 'r', encoding='utf-8') as html_file:
                        content = html_file.read()
                        soup = BeautifulSoup(content, 'html.parser')
                        for message in soup.find_all('div', class_='message default clearfix'):
                            from_name_tag = message.find('div', class_='from_name')
                            current_text_tag = message.find('div', class_='text')
                            if from_name_tag and current_text_tag:
                                reply_to_tag = message.find('div', class_='reply_to details')
                                if reply_to_tag:  # Сообщение является ответом на другое сообщение
                                    reply_to_link = reply_to_tag.find('a')
                                    if reply_to_link:
                                        message_id = reply_to_link['href'].split('go_to_message')[-1]
                                        referenced_message = soup.find('div', id=f'message{message_id}')
                                        if referenced_message:
                                            referenced_text_tag = referenced_message.find('div', class_='text')
                                            if referenced_text_tag:
                                                referenced_text = process_message(referenced_text_tag)
                                                current_text = process_message(current_text_tag)
                                                if contains_cyrillic(referenced_text) and contains_cyrillic(
                                                        current_text) and referenced_text != previous_referenced_text and len(
                                                    referenced_text) > 0 and current_text != referenced_text and referenced_text != previous_message:
                                                    out_file.write(f'<s>{referenced_text} \n')
                                                    previous_referenced_text = referenced_text
                                                if current_text != previous_message and len(
                                                        current_text) > 0 and contains_cyrillic(current_text):
                                                    out_file.write(f'{current_text} \n')
                                                    previous_message = current_text
                                else:  # Сообщение не является ответом
                                    current_text = process_message(current_text_tag)
                                    if contains_cyrillic(current_text) and current_text != previous_message:
                                        out_file.write(f'<s>{current_text} \n')
                                        previous_message = current_text
