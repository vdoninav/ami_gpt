import pymorphy3


# TODO: Сделать morphy глобальным

def load_words(file_path):
    """
    Загружает слова из файла, приводит их к нормальной форме и сохраняет в словарь.
    """
    morph = pymorphy3.MorphAnalyzer()
    word_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip()
            if word:
                # Приводим слово к нормальной форме
                normalized_word = morph.parse(word)[0].normal_form
                word_dict[normalized_word] = True
    return word_dict


def check_word(word_dict, word):
    """
    Проверяет наличие слова в словаре после приведения его к нормальной форме.
    """
    morph = pymorphy3.MorphAnalyzer()
    normalized_word = morph.parse(word)[0].normal_form
    return word_dict.get(normalized_word, False)


# Пример использования
if __name__ == "__main__":
    input_file = 'normalized_words.txt'  # Файл с нормализованными словами

    # Загрузка словаря
    word_dict = load_words(input_file)

    # Пример проверки слов
    words_to_check = ['Привет', 'слово', 'неизвестное_слово']
    for word in words_to_check:
        if check_word(word_dict, word):
            print(f"'{word}' есть в словаре.")
        else:
            print(f"'{word}' нет в словаре.")
