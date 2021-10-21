import re
import requests as req
from bs4 import BeautifulSoup
import math
from fake_useragent import UserAgent
import time
import numpy
import pandas as pd

url = ['http://youngscience.gov.ru/news/news/?page=',
       'https://minobrnauki.gov.ru/press-center/news/?SECTION_ID=59&PAGEN_1=',
       'https://rscf.ru/news/?PAGEN_2='] # используемые ссылки для получения новостей
max_pages = [99, 521, 185] # количество страниц на каждом сайте
Name_tag = [] # все новости с ссылкой на картинку и ссылкой на текст новости
news_count = []
for iter in range(0, 3): # проходим по всем сайтам и по всем страницам на сайте
    for p in range(max_pages[iter]):
        # 1- парсинг сайтов
        img_links = []
        hyp = []
        cur_url = url[iter] + str(p + 1) # страница, на которой мы находимся сейчас
        data = req.get(cur_url, headers={'User-Agent': UserAgent().chrome}) # получаем всю информацию со страницы
        time.sleep(1.0 + numpy.random.uniform(0, 1))
        data_bs = BeautifulSoup(data.text, features="html.parser") #сохраняем весь html код страницы
        print(cur_url) # выводим на экран текущую страницу для наблюдения, где мы находимся
        if iter == 0:
            # Проходим по первому сайту и достаем с него все заголовки новостей, соответствующие картинки и ссылки на новости
            df = data_bs.find_all('strong', class_="news__title")
            img_links = data_bs.find_all('img')
            hyp = data_bs.find_all('div', class_="news")
            for i in range(0, len(hyp)):
                hyp[i] = re.findall(r'<a href=\"(.+)\">', str(hyp[i]))
                hyp[i] = 'http://youngscience.gov.ru' + hyp[i][0]
            for i in range(len(img_links)):
                img_links[i] = 'http://youngscience.gov.ru' + img_links[i].get('src')
                df[i] = df[i].text

        elif iter == 1:
            df = data_bs.find_all('h4', class_="news-item-title")
            img_links = data_bs.find_all('img', loading="lazy")
            for i in range(0, len(df)):
                hyp.append(re.findall(r'<a href=\"(.+)\">', str(df[i])))
                hyp[i] = 'https://minobrnauki.gov.ru' + hyp[i][0]
            for i in range(0, len(img_links)):
                img_links[i] = 'https://minobrnauki.gov.ru' + img_links[i].get('src')
                df[i] = df[i].text

        elif iter == 2:
            df = data_bs.find_all(class_="news-title")
            img_links = data_bs.find_all('img', alt="")
            del img_links[:6]
            img_links.pop()
            for i in range(len(img_links)): # нет отдельного тега для href
                img_links[i] = 'https://rscf.ru' + img_links[i].get('src')
                hyp.append('https://rscf.ru' + df[i].get('href'))
                df[i] = df[i].text

        for i, j, k in zip(df, img_links, hyp): # отбираем новости по определенным токенам и добавляем в Name_tag вместе со
            # всеми соответствующими атрибутами
            name_tag = i
            image_tag = j
            hyper_tag = k
            if name_tag.find('молод') > 0 or name_tag.find('аспир') > 0 or name_tag.find('студ') > 0:
                Name_tag.append((''.join(name_tag), image_tag, hyper_tag))
    news_count.append(len(Name_tag)) # в news_count будут храниться три числа.
    # В первом - количество новостей на первом сайте, во втором - количество на первом и на втором сайте,
    # третье - общее количество нужных новостей.

# 2- запись в словарь
dictionary_all = {}
tf = []
cur = 0
pd.DataFrame({'url': ['url'], 'text': ['text']}).to_csv('my_news.csv', index=False, header=False)  # в файле эксель создаем таблицу с ссылкой и новостью
for i in Name_tag:
    cur += 1
    dictionary = {}
    url = i[2]
    print(url)
    news_data = req.get(url, headers={'User-Agent': UserAgent().chrome})
    time.sleep(1.0 + numpy.random.uniform(0, 1))
    newsData_bs = BeautifulSoup(news_data.text, features="html.parser")
    # достаем контент со страницы в зависимости от сайта
    if cur <= news_count[0]:
        content = newsData_bs.find_all('div', class_="l-col1")
    elif cur <= news_count[1]:
        content = newsData_bs.find_all('div', class_="post-body")
    else:
        content = newsData_bs.find_all(class_="b-news-detail-content")

    page = ''
    words = 0
    # переводим текст новости из массива в строку
    for i in range(0, len(content)):
        page += content[i].text + ' '
    pd.DataFrame({'url': [url], 'text': [page]}).to_csv('my_news.csv', index=False, header=False, mode='a')  # записываем новость в эксель

    # tf
    for i in re.findall(r'\b[a-zA-zА-ЯЁа-яё]+\b', page):  # просматриваем все слова в новости
        i = i.lower()
        dictionary[i] = dictionary.setdefault(i, 0) + 1 # в dictionary будем хранить в дальнейшем tf
        dictionary_all[i] = dictionary_all.setdefault(i, 0) + 1 # в dictionary_all - количество употребления слова
        words = words + 1
    for k, v in dictionary.items():
        dictionary[k] = v / words
    tf.append(dictionary)

idf = {}
for k, v in dictionary_all.items():
    vh = 0  # число документов, содержащих слово
    for i in tf: # проходим по всем документам, если в текущем документе мы встретили рассматриваемое слово, то
        # увеличиваем число документов, содержащих слово
        if k in i:
            vh += 1
    idf[k] = math.log(len(Name_tag) / vh)

tfidf = []
for i in tf:
    cur_tfidf = {}
    for k, v in i.items():
        cur_tfidf[k] = v * idf[k]
    tfidf.append(cur_tfidf)
for cur in tfidf:
    print(sorted(cur.items(), key=lambda x: x[1], reverse=True)[:5]) # выводим топ 5 слов по tf-idf для каждой новости
sorted_dict = sorted(dictionary_all.items(), key=lambda x: x[1],reverse=True)  # сортируем количество слов в документах по убыванию
sorted_dict = dict(sorted_dict)

# вывод самых популярных слов
import pymorphy2

dict_show = {}


def pos(word, morth=pymorphy2.MorphAnalyzer()): # функция для определения части речи слова
    return morth.parse(word)[0].tag.POS


functors_pos = {'INTJ', 'PRCL', 'CONJ', 'PREP', 'NPRO'} # ненужные части речи

for k, v in sorted_dict.items(): # отбираем слова по количеству вхождений и части речи
    if v >= 500 and pos(k) not in functors_pos:
        dict_show[k] = v

import matplotlib.pyplot as plt

plt.bar(dict_show.keys(), dict_show.values()) # выводим график на экран
plt.show()

# 3 - вывод в html
file_name = "my.html"
text = "<html><title></title><head></head><body>"
f = open(file_name, "w")
f.write(text)
f.close()

for i in Name_tag:
    url = i[0]
    src = i[1]
    ssl = i[2]
    with open(file_name, 'a', encoding='utf-8') as out_file:
        out_file.write(
            "<figure class=""image"">  <img src=" + src + " width = ""400"" hight = ""400""><figcaption><a href=" + ssl + ">" + url + "</a></figcaption></figure>")

f = open(file_name, "a")
f.write("</body></html>")
