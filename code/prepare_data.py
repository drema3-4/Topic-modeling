import pandas as pd
import re

!pip install nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

!pip install pymorphy2
import pymorphy2

# Загрузка данных и отсечение последней строки с несущественными столбцами (link, data, tags)
news = pd.read_excel('news.xlsx')
news = news[:-1]
news = news[['title', 'content']]

# Функция для разбиения ячеек на слова
def tokenize(cell: str) -> list[str]:
    words = []

    sentences = nltk.sent_tokenize(cell)
    for sentence in sentences:
        words += nltk.word_tokenize(sentence)

    return words

# Функция для перевода слов в нижний регистр
def convert_to_lowercase(words: list[str]) -> list[str]:
    new_words = []

    for word in words:
        new_words.append(word.lower())

    return new_words

# Функция для удаления символов, отличающихся от символов русского и английского алфавитов
def del_non_alphs(words: list[str]) -> list[str]:
    new_words = []

    for word in words:
        new_word = ''

        for symbol in word:
            if (symbol >= 'a' and symbol <= 'z' or symbol >= 'а' and symbol <= 'я'):
                new_word += symbol

        if (len(new_word) > 0):
            new_words.append(new_word)

    return new_words

# Функция для удаления стоп слов
def del_stop_words(words: list[str]) -> list[str]:
    new_words = []

    for word in words:
        if re.match('[а-я]', word):
            if word not in (stopwords.words('russian') + ['вшэ' + 'ниу']):
                new_words.append(word)
        elif re.match('[a-z]', word):
            if word not in stopwords.words('english'):
                new_words.append(word)

    return new_words

# Функция лемматизации
def lemm_words(words: list[str]) -> list[str]:
    lemm_nltk = WordNetLemmatizer()
    lemm_pymorphy2 = pymorphy2.MorphAnalyzer()

    new_words = []

    for word in words:
        if re.match('[а-я]', word):
            new_words.append(lemm_pymorphy2.parse(word)[0].normal_form)
        elif re.match('[a-z]', word):
            new_words.append(lemm_nltk.lemmatize(word))

    return new_words

# Функция для конвертации массива строк в предложение
def convert_words_to_cell(words: list[str]) -> str:
    cell = ' '.join(words)

    return cell

# Функция для применения остальных функция предобработки
def colaider(data: pd.DataFrame) -> None:
    for column in ['title', 'content']:
        for cell in range(data.shape[0]):
            temp = data[column].loc[cell]

            words = tokenize(temp)
            words = convert_to_lowercase(words)
            words = del_non_alphs(words)
            words = del_stop_words(words)
            words = lemm_words(words)
            temp = convert_words_to_cell(words)

            data.loc[cell, column] = temp

# Выполнение предобработки
colaider(news)

# Функция для удаления пустых строк массива
def del_void_string(data: pd.DataFrame) -> None:
    for string in range(data.shape[0]):
        if len(data.loc[string, 'title']) == 0 and len(data.loc[string, 'content']) == 0:
            data = data.drop(string)

# Удаление пустых строк
del_void_string(news)

# Сохраняем результаты
news.to_excel('prepeared_news.xlsx')