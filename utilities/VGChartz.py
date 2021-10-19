import requests
from bs4 import BeautifulSoup
from rich import print
import pandas as pd
from rich.progress import track
import re
import numpy as np
from os.path import exists
import ast
import time
from fake_useragent import UserAgent

URL = 'https://www.vgchartz.com/games/games.php?page=1&results=200&order=Sales&ownership=Both&direction=DESC&showtotalsales=1&shownasales=1&showpalsales=1&showjapansales=1&showothersales=1&showpublisher=1&showdeveloper=1&showreleasedate=1&showlastupdate=1&showvgchartzscore=1&showcriticscore=1&showuserscore=1&showshipped=1&showmultiplat=Yes'
HEADERS = ['index', 'img', 'Game', 'Console', 'Publisher', 'Developer', 'VGChartz Score', 'Critic Score',
          'User Score', 'Total Shipped', 'Total Sales', 'NA Sales', 'PAL Sales', 'Japan Sales', 'Other Sales',
          'Release Date', 'Last Update', 'ID']

UA = UserAgent()
def request_url(url: str) -> BeautifulSoup:
    """
    Takes website URL and return soup object by using different User-Agent everytime you make a request.
    Parameters
    ----------
    url: it's a sting that hold the Uniform Resource Locators aka URL.

    Returns
    ----------
    BeautifulSoup object that contain the html of the given url.
    """
    global THREAD_ID
    headers = {'User-Agent': UA.random}
    try:
        response = requests.get(url, headers=headers)
        if response != 200:
            time.sleep(2)
            return request_url(url)
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
        print(type(e).__name__)
        time.sleep(2)
        return request_url(url)
    soup = BeautifulSoup(response.content, "html5lib")
    return soup

def clean_data(data):
    w = data.text.strip()
    if w == '':
        return data.find('img')['alt'].strip()
    return w

def get_games_details(id:int)-> dict:
    soup = request_url(f'https://www.vgchartz.com/game/{id}/super-mario-galaxy/?region=America')
    left_column = soup.find('div', id='left_column')
    try:
        genre = left_column.find('div', id='gameGenInfoBox').find(text="Genre").findNext('p').contents[0]
    except Exception:
        print(f'Error at {id}')
    return {'ID': id,'Genre' : genre}

def scrap(soap: BeautifulSoup = None) -> list:
    """
    Parameters
    ----------
    soap : a bs4.BeautifulSoup

    Returns
    ----------
    List
        """
    lst = []
    lst1 = []
    for i in soap:
        dic = {}
        for index, data in enumerate(i.find_all('td')):
            if data.find('a'):
#                 print(data.find('a'))
                dic['ID'] = int(re.findall('\d+', data.find('a')['href'].strip())[0])
            dic[HEADERS[index]] = clean_data(data)
#         print(dic)
        lst.append(dic)
    return lst[2:]

def get_games_data(start_page:int=1, end_page:int=306):
    if not exists('data/VGChartz.csv'):
        df_list = list()
        for page in range(start_page, end_page):
            soap_ = request_url(
                f'https://www.vgchartz.com/games/games.php?page={page}&results=200&order=VGChartzScore&ownership=Both&showtotalsales=1&shownasales=1&showpalsales=1&showjapansales=1&showothersales=1&showpublisher=1&showdeveloper=1&showreleasedate=1&showlastupdate=1&showvgchartzscore=1&showcriticscore=1&showuserscore=1&showshipped=1&showmultiplat=Yes').find('div', id='generalBody').find('table').find_all('tr')[1:]
            descr = scrap(soap_)
            df_list.append(descr)
        df = pd.DataFrame([r for d in df_list for r in d]).reset_index(drop=True)[HEADERS[2:]]
        lst = list()
        for i in df.ID:
            lst.append(get_games_details(i))
        return df.merge(pd.DataFrame(lst), on='ID')
    else:
        df = pd.read_csv('data/VGChartz.csv').drop('Unnamed: 0', axis=1)
        return df
