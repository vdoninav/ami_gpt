import os
import regex
from bs4 import BeautifulSoup


def process_message(text):
    # Очищаем текст, оставляя только кириллицу, пробелы, знаки препинания и цифры
    cleaned_text = regex.sub(r'[^\p{Cyrillic}\s.,?!-\d]', '', text.get_text(strip=True))
    # Убираем последовательности из четырех и более цифр
    cleaned_text = regex.sub(r'\d{4,}', '', cleaned_text)
    # Убираем лишние пробелы
    cleaned_text = regex.sub(r'\s+', ' ', cleaned_text)
    # Убираем повторяющиеся знаки препинания
    cleaned_text = regex.sub(r'([ !,.?])\1+', ' ', cleaned_text)
    return cleaned_text


def contains_cyrillic(input_string):
    return bool(regex.search(r'\p{IsCyrillic}', input_string))


directory = 'htmls'  # Where telegram exports lie
output_file = 'texts/output_all_6_with_nums.txt'

previous_message = ""  # Add lines to store previous messages
previous_referenced_text = ""  # Add lines to store previous referenced texts

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
                                if reply_to_tag:  # Message is a reply to another message
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
                                                    out_file.write(f'{referenced_text} \n')
                                                    previous_referenced_text = referenced_text
                                                if current_text != previous_message and len(
                                                        current_text) > 0 and contains_cyrillic(current_text):
                                                    out_file.write(f'{current_text} \n')
                                                    previous_message = current_text
                                else:  # Message is not a reply
                                    current_text = process_message(current_text_tag)
                                    if contains_cyrillic(current_text) and current_text != previous_message:
                                        out_file.write(f'{current_text} \n')
                                        previous_message = current_text
