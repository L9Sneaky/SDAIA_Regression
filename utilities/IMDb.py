import requests
from bs4 import BeautifulSoup
from rich import print
import pandas as pd
from rich.progress import track
import re
import numpy as np
from os.path import exists
import ast


def request_url(url: str) -> BeautifulSoup:
    """
    Takes website URL and return soup object
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html5lib")
    return soup


def scrap(soap: BeautifulSoup = None) -> list:
    """
    Parameters
    ----------
    soap : a bs4.BeautifulSoup

    Returns
    ----------
    List
        """
    lst = list()
    lst2 = list()
    games = soap.find('div', class_='lister-list').find_all('div',
                                                            class_='lister-item-content')
    for game in games:
        dic = dict()
        try:
            dic['Title'] = game.find('a').text.strip()
        except AttributeError:
            dic['Title'] = np.nan
        try:
            dic['Year'] = int(re.findall('\d+', game.find('h3').find('span',
                              class_='lister-item-year text-muted unbold').text)[0])
        except (AttributeError, IndexError) as e:
            dic['Year'] = np.nan
        try:
            dic['id'] = game.find('a')['href'].split('/')[2]
        except AttributeError:
            dic['id'] = np.nan
        try:
            dic['Genre'] = game.find(
                'span', class_='genre').text.strip().split(', ')
        except AttributeError:
            dic['Genre'] = np.nan
        try:
            dic['Rating'] = float(
                game.find('div', class_='ratings-bar').find('strong').text)
        except AttributeError:
            dic['Rating'] = np.nan
        try:
            dic['Director'] = game.find_all('p')[2].a.text
        except AttributeError:
            dic['Director'] = np.nan
        try:
            dic['Votes'] = int(
                game.find('span', attrs={'name': 'nv'}).text.strip().replace(',', ''))
        except AttributeError:
            dic['Votes'] = np.nan
        lst.append(dic)
        lst2.append(get_rating(dic['id']))
    return lst, lst2


def rating_helper(df: pd.DataFrame()):
    header = ['All', '<18', '18-29', '30-44', '45+']
    df = df.T[header]
    df.columns = pd.MultiIndex.from_product([df.index, df.columns])
    return df.reset_index(drop=True)


def get_rating(id: str):
    page = request_url(
        f'https://www.imdb.com/title/{id}/ratings/?ref_=tt_ov_rt')

    table = page(text='Rating By Demographic')[
        0].parent.find_next_sibling().find_next_sibling()
    lst = list()
    header = ['All', '<18', '18-29', '30-44', '45+']
    for i in range(1, 4):
        inner_dict = dict()
        for k, j in enumerate(table.find_all('tr')[i].find_all('td', class_='ratingTable')):
            if j.find('a') != None:
                inner_dict[header[k]] = j.find('a').text.strip()
            else:
                inner_dict[header[k]] = np.nan
        lst.append(inner_dict)
        # lst.append({j:x for x, j in zip(table.find_all('tr')[i].find_all('td', class_='ratingTable'), header)})
    total = pd.DataFrame.from_dict(({'All': lst[0]}))
    male = pd.DataFrame({'Male': lst[1]})
    female = pd.DataFrame({'Female': lst[2]})
    total = rating_helper(total)
    male = rating_helper(male)
    female = rating_helper(female)
    result = total.join([male, female])
    return result


def try_convert_to_list(value):
    try:
        x = ast.literal_eval(value)
        return x
    except ValueError:
        return np.nan


def get_games_data():
    if not exists('data/imbd_games2.csv'):
        df_list = list()
        rating_df_list = list()
        for start in track(range(1, 9951, 50), description="[red]Scrapping..."):
            soap_ = request_url(
                f'https://www.imdb.com/search/title/?title_type=video_game&sort=user_rating,desc&start={start}&ref_=adv_nxt')
            descr, rating_df = scrap(soap_)
            df_list.append(descr)
            rating_df_list.append(rating_df)
        # return pd.concat([r for d in rating_df_list for r in d]).reset_index(drop=True)
        return pd.DataFrame([r for d in df_list for r in d]).join(pd.concat([r for d in rating_df_list for r in d]).reset_index(drop=True))
    else:
        df = pd.read_csv('data/imbd_games2.csv').drop('Unnamed: 0', axis=1)
        df['Genre'] = df['Genre'].apply(lambda x: try_convert_to_list(x))
        return df
