import pymorphy3


def normalize_words(input_file, output_file):
    # Инициализация pymorphy2
    morph = pymorphy3.MorphAnalyzer()

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Удаление лишних пробелов и перевод слова к нормальной форме
            word = line.strip()
            if word:
                normalized_word = morph.parse(word)[0].normal_form
                outfile.write(normalized_word + '\n')


if __name__ == "__main__":
    input_file = 'rus.txt'  # Имя файла со словами
    output_file = 'rus_norm.txt'  # Имя файла для записи нормализованных слов
    normalize_words(input_file, output_file)
