import os
import pandas as pd
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # используем, чтобы не выдавались предупреждения

df = pd.read_csv('lenta-ru-news.csv', delimiter=',')[['text', 'tags']] # берем информацию с готового датасета
labels = df.groupby('tags').count()  # количество новостей для категории
labels = labels.rename(columns={'text': 'cnt'})
labels = labels[(6000 < labels.cnt) & (labels.cnt < 100000)]
df = pd.merge(df, labels, on='tags')  # объединяем таблицы по категориям
print(df.tags.value_counts()) # выводим отсортированные по убыванию
topics = ['Бизнес', 'Госэкономика', 'Интернет', 'Кино', 'Люди', 'Музыка', 'Наука', 'Общество', 'Политика',
          'Происшествия', 'Следствие и суд', 'Украина', 'Футбол']

MAX_NB_WORDS = 30000  # топ 30000 самых встречаемых слов для обучения
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 32


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\\n", '', text)  # убираем все \n
    text = re.sub(r'[/(){}\[\]\|@,;]', ' ', text)
    text = re.sub(r'[^0-9а-я #+_]', '', text)  # для уверенности в том, что удалились все посторонние символы, в том числе иностранные буквы
    return text

df = df.reset_index(drop=True)  # убираем колонку индексов
df['text'] = df['text'].apply(clean_text)  # заменяем исходный текст очищенным текстом

# Создаем экземпляр токенизатора tokenizer. В качестве аргумента передадим параметр num_words, содержащий максимальное количество
# слов в словаре токенайзера. Если в подлежащем токенизации тексте окажется больше слов,
# то в словарь
# будут помещены  слов, которые встречаются в тексте чаще остальных.
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
tokenizer.fit_on_texts(df['text'].values) #  Это позволит токенайзеру обойти весь текст и обновить словарь токенов в соответствии с частотой вхождения слов,
#  так, чтобы наиболее популярные слова в тексте получили наименьшие индексы.

# Применяем полученный шаблон к текстам и категориям
X = tokenizer.texts_to_sequences(df['text'].values)  # разбиваем статью на слова и слова превращаем в числа
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)  # подгоняем новость под одну длину
Y = pd.get_dummies(df['tags']).values  # некая матрица - 12 нулей, 1 единица с темой новости

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10,
                                                    random_state=42)  # разделяем на тестовые и тренировочные
# random state для перемешивания экземпляров, чтобы не возникало предвзятости

model = tf.keras.models.Sequential() # создаем модель
model.add(tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)) # задаем первый слой модели
model.add(tf.keras.layers.LSTM(100))  # добавляем 100 слоёв реккурентной LSTM
model.add(tf.keras.layers.Dense(len(labels), activation='softmax'))  # Добавим слой softmax с len(labels) выходами
# Softmax преобразует вектор значений в распределение вероятностей.
# Softmax часто используется в качестве активации для последнего слоя классификационной сети,
# поскольку результат может быть интерпретирован как распределение вероятности.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # настраиваем процесс обучения с помощью
# compile.
model.fit(X_train, Y_train, epochs=3, batch_size=64) # обучаем модель на тренировочных данных
# 3 итерации по всем входным данным, При передаче данных NumPy, модель разбивает данные на меньшие блоки (batches)
# и итерирует по этим блокам во время обучения. Это число указывает размер каждого блока данных.
#  последний блок может быть меньшего размера если общее число записей не делится на размер партии.

df_class = pd.read_csv('my_news.csv')  # массив наших новостей(ссылка и текст столбцами)
for article in df_class.values.tolist():  # article - массив, в котором лежат ссылка и текст
    url = article[0]
    text_n = ' ' + article[1]
    text_n = clean_text(text_n)
    text_n = re.sub(r'\d+', '', text_n)
    seq = tokenizer.texts_to_sequences([text_n])  # числовое представление техта
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)  # подгоняем под общую длину(убеждаемся, что имеют одну длину)
    pred = model.predict(padded)  # Количество элементов в массиве - количество категорий, значение - вероятность попадания в категорию.
    print(topics[np.argmax(pred)], url)
