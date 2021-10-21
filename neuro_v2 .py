import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv('lenta-ru-news.csv', delimiter=',')[['text', 'tags']]
labels = df.groupby('tags').count()  # количество новостей для категории
labels = labels.rename(columns={'text': 'cnt'})
labels = labels[(6000 < labels.cnt) & (labels.cnt < 100000)]
df = pd.merge(df, labels, on='tags')  # объединяем таблицы по категориям
print(df.tags.value_counts())
topics = ['Политика', 'Общество', 'Украина', 'Происшествия', 'Госэкономика', 'Футбол', 'Кино', 'Интернет', 'Бизнес',
          'Следствие и суд', 'Наука', 'Музыка', 'Люди']
MAX_NB_WORDS = 30000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 32

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\\n', '', text)
    text = re.sub(r'[/(){}\[\]\|@,;]', ' ', text)
    text = re.sub(r'[^0-9а-я #+_]', '', text)
    return text

df = df.reset_index(drop=True)
df['text'] = df['text'].apply(clean_text)  # заменяем исходный текст очищенным текстом

# разбиение текста на слова, все в нижнем регистре без всякой фигни по типу знаков препинания
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
tokenizer.fit_on_texts(df['text'].values)

X = tokenizer.texts_to_sequences(df['text'].values)  # статья, разбитая на слова
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)  # подгоняем новость под одну длину
Y = pd.get_dummies(df['tags']).values  # массив всех наших категорий
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)  # разделяем на тестовые и тренировочные

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.LSTM(100))  # поясняем, что как бы работаем с реккурентной сетью
model.add(tf.keras.layers.Dense(len(labels), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=3, batch_size=64)

df_class = pd.read_csv('my_news.csv')  # массив наших новостей(ссылка и текст столбцами)
for article in df_class.values.tolist():  # article - массив, в котором лежат ссылка и текст
    url = article[0]
    article = clean_text(article[1])
    article = re.sub(r'\d+', '', article)  # удаляем циферки
    seq = tokenizer.texts_to_sequences([article])  # разбиваем текст на токены
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)  # подгоняем под общую длину
    pred = model.predict(padded)  # Количество элементов в массиве - количество категорий, значение - вероятность попадания в категорию.
    print(topics[np.argmax(pred)], url)
