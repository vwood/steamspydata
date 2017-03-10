#!/usr/bin/env python3

import numpy as np
from zipfile import ZipFile
import pandas as pd
import scipy as sp
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.linear_model import LinearRegression, LogisticRegression
import myutils as mu
import pylab as pl
import glob
import seaborn as sns

"""
Analysis:

Every plot here is a log plot on at least one axis
Games sales are distributed very exponentially, with decreasing numbers of games at increasing levels of profitability. There is a limited amount of brand space to go around on the Steam platform.

TODO - look into release dates (will need months, sigh)

TODO - remove free games (skew results because they don't make profit at all!)


"""


d1 = pd.read_csv('data/2016 Games on Steam - Indie AND Action.csv').dropna(how='all')
d1.rename(columns={'#': '#_indie&action'}, inplace=True)
d1['is_action'] = 1
d1['is_indie'] = 1

d2 = pd.read_csv('data/2016 Games on Steam - Indie AND Strategy.csv').dropna(how='all')
d2.rename(columns={'#': '#_indie&strategy'}, inplace=True)
d2['is_strategy'] = 1
d2['is_indie'] = 1

d3 = pd.read_csv('data/2016 Games on Steam - Indie AND Sandbox.csv').dropna(how='all')
d3.rename(columns={'#': '#_indie&sandbox'}, inplace=True)
d3['is_sandbox'] = 1
d3['is_indie'] = 1

d4 = pd.read_csv('data/2016 Games on Steam - Indie.csv').dropna(how='all')
d4.rename(columns={'#': '#_indie'}, inplace=True)
d4['is_indie'] = 1

d5 = pd.read_csv('data/2016 Games on Steam - Indie AND Survival.csv').dropna(how='all')
d5.rename(columns={'#': '#_indie&survival'}, inplace=True)
d5['is_survival'] = 1
d5['is_indie'] = 1

d6 = pd.read_csv('data/2016 Games on Steam - Indie AND Simulation.csv').dropna(how='all')
d6.rename(columns={'#': '#_indie&simulation'}, inplace=True)
d6['is_simulation'] = 1
d6['is_indie'] = 1

d7 = pd.read_csv('data/2016 Games on Steam - Platformer AND Puzzle.csv').dropna(how='all')
d7.rename(columns={'#': '#_platformer&puzzle'}, inplace=True)
d7['is_platformer'] = 1
d7['is_puzzle'] = 1

d8 = pd.read_csv('data/2016 Games on Steam - Indie AND Platformer.csv').dropna(how='all')
d8.rename(columns={'#': '#_indie&platformer'}, inplace=True)
d8['is_platformer'] = 1
d8['is_indie'] = 1

d9 = pd.read_csv('data/2016 Games on Steam - Indie AND RPG.csv').dropna(how='all')
d9.rename(columns={'#': '#_indie&RPG'}, inplace=True)
d9['is_RPG'] = 1
d9['is_indie'] = 1

for df in [d1, d2, d3, d4, d5, d6, d7, d8, d9]:
    df.drop('c', axis=1, inplace=True)
    if 'Owners X Price' in df:
        df.drop('Owners X Price', axis=1, inplace=True)
    if 'Price X Owners' in df:
        df.drop('Price X Owners', axis=1, inplace=True)


all_games = pd.Series(list(set(d1['Game'])
                           .union(set(d2['Game']))
                           .union(set(d3['Game']))
                           .union(set(d4['Game']))
                           .union(set(d5['Game']))
                           .union(set(d6['Game']))
                           .union(set(d7['Game']))
                           .union(set(d8['Game']))
                           .union(set(d9['Game']))))

all_columns = list(set(d1.columns)
                   .union(set(d2.columns))
                   .union(set(d3.columns))
                   .union(set(d4.columns))
                   .union(set(d5.columns))
                   .union(set(d6.columns))
                   .union(set(d7.columns))
                   .union(set(d8.columns))
                   .union(set(d9.columns)))
all_columns.remove('Game')

results = dict((column, {}) for column in all_columns)
for df in [d1, d2, d3, d4, d5, d6, d7, d8, d9]:
    for i, row in df.iterrows():
        game = row['Game']
        for col in all_columns:
            if col in row and not pd.isnull(row[col]):
                results[col][game] = row[col]


data = {'Game':all_games}
for column, result in results.items():
    this_column = []
    for game in all_games:
        this_column.append(result.get(game, np.nan))

    data[column] = this_column

data = pd.DataFrame(data)

def convert_number(s):
    if s == 'Free': return 0.0
    return float(s.replace('$', '').replace(',', ''))
def convert_time(s):
    result = 0
    for i in s.split(':'):
        result *= 60
        result += int(i)
    return result
        
data['Price'] = data['Price'].apply(convert_number)
data['Owners'] = data['Owners'].apply(convert_number)
data['Players'] = data['Players'].apply(convert_number)
data['Median Playtime'] = data['Median Playtime'].apply(convert_time)
data['Avg Playtime'] = data['Avg Playtime'].apply(convert_time)

for c in ['is_RPG', 'is_action', 'is_indie', 'is_platformer', 'is_puzzle',
          'is_sandbox', 'is_simulation', 'is_strategy', 'is_survival']:
    data[c].fillna(0, inplace=True)


def score_split(s):
    s = (s.replace('N/A', '!')
         .replace('(','')
         .replace(')','')
         .replace('/', ' ')
         .replace('%',''))
    s = s.split(' ')

    return [np.nan if i == '!' else int(i) for i in s]
    
data['Score'] = data['Score rank(Userscore / Metascore)'].apply(lambda x:
                                                                score_split(x)[0])
data['UserScore'] = data['Score rank(Userscore / Metascore)'].apply(lambda x:
                                                                    score_split(x)[1])
#data['MetaScore'] = data['Score rank(Userscore / Metascore)'].apply(lambda x:
#                                                                    score_split(x)[2])

data['HasScore'] = (pd.isnull(data['Score']) == False).astype(int)
data['HasUserScore'] = (pd.isnull(data['UserScore']) == False).astype(int)
#data['HasMetaScore'] = (pd.isnull(data['MetaScore']) == False).astype(int)
data['HasScore>50%'] = (data['Score'] > 50).astype(int)
data['HasScore>60%'] = (data['Score'] > 60).astype(int)
data['HasScore>70%'] = (data['Score'] > 70).astype(int)
data['HasScore>80%'] = (data['Score'] > 80).astype(int)
data['HasScore>90%'] = (data['Score'] > 90).astype(int)
data['HasScore>90%'] = (data['Score'] > 90).astype(int)

data['Approx Profit'] = data['Owners'] * data['Price']
# data['2016-ReleaseYear'] = data['Release date'].apply(lambda x:2016-int(x.split('/')[-1]))

pl.scatter(data['Price'],
           np.log(data['Approx Profit']),
           alpha=0.3, lw=0.2)
pl.xlabel("Price")
pl.ylabel("log Profit")
mu.plot_out()

pl.scatter(np.log(data['Players']),
           np.log(data['Approx Profit']), alpha=0.3, lw=0.2)
pl.xlabel("log Players")
pl.ylabel("log Profit")
mu.plot_out()

pl.scatter(np.log(data['Median Playtime']),
           np.log(data['Approx Profit']),
           alpha=0.3, lw=0.2)
pl.xlabel("log Median Playtime")
pl.ylabel("log Profit")
mu.plot_out()

pl.scatter(np.log(data['Avg Playtime']),
           np.log(data['Approx Profit']),
           alpha=0.3, lw=0.2)
pl.xlabel("log Avg Playtime")
pl.ylabel("log Profit")
mu.plot_out()

pl.scatter(np.log(data['Players']),
           np.log(data['Owners']),
           alpha=0.3, lw=0.2)
pl.xlabel("log Players")
pl.ylabel("log Owners")
mu.plot_out()


features = ['Avg Playtime', 'Median Playtime', 'Players', 'is_RPG', 'is_action', 'is_indie', 'is_platformer', 'is_sandbox', 'is_simulation', 'is_strategy', 'is_survival', 'HasScore', 'HasScore>50%']
xs = data[features].values
ys = data['Approx Profit'].values
m = LinearRegression()
m.fit(xs, ys)

print "R^2: {:8.4f}".format(m.score(xs, ys))
print
for feature, coef in zip(features, m.coef_):
    print "{:20s} {:16.4f}".format(feature, coef)
print "{:20s} {:16.4f}".format("intercept", m.intercept_)
print

if True:
    print data.ix[0]

print
print data['Release date']
    

