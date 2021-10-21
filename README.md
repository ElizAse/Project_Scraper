#Project_Scraper

В данном проекте мы создавали скрапер новостей, касающихся молодежной политики, студентов и аспирантов на основе токенов: «молод, аспир, студ». 
Полученные новости структурировали на html странице в порядке: картинка – текст. 
Ссылки для скрапинга: http://youngscience.gov.ru/news/news/ https://minobrnauki.gov.ru/press-center/ https://rscf.ru/news/
Также мы составили словарь уникальных слов в тексте публикаций, отфильтровали предлоги/союзы и остальные служебные части речи и изобразили график частотности слов. 
Далее мы производили замеры важности слов на основе TF-IDF меры и применили рекуррентную сеть для классификации тематики новостей.

Для начала нужно запустить файл main.py, в котором создастся html страница и csv файл.
Затем для запуска neuro.py требуется скачать датасет https://www.kaggle.com/yutkin/corpus-of-russian-news-articles-from-lenta и поместить датасет в папку с проектом.

Используемые источники при работе над проектом: https://habr.com/ru/post/280238/ , https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
Используемые библиотеки: bs4, requests, time, fake_useragent, pandas, re, pymorphy2, tensorflow, numpy, matplotlib.
Язык: Python3. 
