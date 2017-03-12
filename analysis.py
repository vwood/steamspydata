#!/usr/bin/env python3

from __future__ import print_function
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
from collections import Counter
from scipy.stats import boxcox

pl.rc('font', size=16)
pl.rc('axes', labelsize=16)
pl.rc('xtick', labelsize=12)
pl.rc('ytick', labelsize=12)
"""
Analysis:

Every plot here is a log plot on at least one axis
Games sales are distributed very exponentially, with decreasing numbers of games at increasing levels of profitability. There is a limited amount of brand space to go around on the Steam platform.

TODO - examine outliers, add classes to graphs, annotate with game names, draw on top of graphs with clusters

TODO - check outliers that he mentions in presentation (small number of players causing instability in playtime variables?)
"""


d1 = pd.read_csv('data/2016 Games on Steam - Indie AND Action.csv').dropna(how='all')
d1.rename(columns={'#': '#_indie&action'}, inplace=True)
d1['is_action'] = 1

d2 = pd.read_csv('data/2016 Games on Steam - Indie AND Strategy.csv').dropna(how='all')
d2.rename(columns={'#': '#_indie&strategy'}, inplace=True)
d2['is_strategy'] = 1

d3 = pd.read_csv('data/2016 Games on Steam - Indie AND Sandbox.csv').dropna(how='all')
d3.rename(columns={'#': '#_indie&sandbox'}, inplace=True)
d3['is_sandbox'] = 1

d4 = pd.read_csv('data/2016 Games on Steam - Indie.csv').dropna(how='all')
d4.rename(columns={'#': '#_indie'}, inplace=True)

d5 = pd.read_csv('data/2016 Games on Steam - Indie AND Survival.csv').dropna(how='all')
d5.rename(columns={'#': '#_indie&survival'}, inplace=True)
d5['is_survival'] = 1

d6 = pd.read_csv('data/2016 Games on Steam - Indie AND Simulation.csv').dropna(how='all')
d6.rename(columns={'#': '#_indie&simulation'}, inplace=True)
d6['is_simulation'] = 1

d7 = pd.read_csv('data/2016 Games on Steam - Platformer AND Puzzle.csv').dropna(how='all')
d7.rename(columns={'#': '#_platformer&puzzle'}, inplace=True)
d7['is_platformer'] = 1
d7['is_puzzle'] = 1

d8 = pd.read_csv('data/2016 Games on Steam - Indie AND Platformer.csv').dropna(how='all')
d8.rename(columns={'#': '#_indie&platformer'}, inplace=True)
d8['is_platformer'] = 1

d9 = pd.read_csv('data/2016 Games on Steam - Indie AND RPG.csv').dropna(how='all')
d9.rename(columns={'#': '#_indie&RPG'}, inplace=True)
d9['is_RPG'] = 1

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


for c in ['is_RPG', 'is_action', 'is_puzzle', 'is_platformer', 'is_puzzle',
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

data['Approx Net Revenue'] = data['Owners'] * data['Price']

# Drop nan dates - these are a few games from pre-2016
data.dropna(subset=['Release date'], inplace=True)

# data['Original date'] = data['Release date']
data['Release date'] = pd.to_datetime(data['Release date'])

# Confirm all dates are from 2016
# print(max(data['Release date']))

end_of_2016 = pd.to_datetime('2017-01-01')
data['DaysSinceRelease'] = data['Release date'].apply(lambda x: (end_of_2016 - x).days)

data['VRInTitle'] = data['Game'].apply(lambda x:
                                       int('vr' in x.lower()))
data['2InTitle'] = data['Game'].apply(lambda x:
                                      int('2' in x))
data['SpaceInTitle'] = data['Game'].apply(lambda x:
                                          int('space' in x.lower()))
data['SuperInTitle'] = data['Game'].apply(lambda x:
                                          int('super' in x.lower()))
data['TheInTitle'] = data['Game'].apply(lambda x:
                                        int('the' in x.lower()))
data['OfInTitle'] = data['Game'].apply(lambda x:
                                       int('of' in x.lower()))

data = data[data['Price'] > 0.0]

figsize = (10, 7)
dpi = 100

pl.figure(figsize=figsize, dpi=dpi)
pl.semilogy(data['Price'],
            data['Approx Net Revenue'],
            'bo',
            alpha=0.3, lw=0.2)
pl.xlabel("Price")
pl.ylabel("Estimated Net Revenue (log)")
mu.plot_out()

pl.figure(figsize=figsize, dpi=dpi)
pl.hexbin(data['Price'],
          np.log(data['Approx Net Revenue']),
          gridsize=32,
          mincnt=1,
          cmap=pl.cm.winter)
pl.colorbar()          
pl.xlabel("Price")
pl.ylabel("Estimated Net Revenue (log)")
mu.plot_out()


pl.figure(figsize=figsize, dpi=dpi)
pl.loglog(data['Players'],
          data['Approx Net Revenue'],
          'bo',
          alpha=0.3, lw=0.2)
pl.xlabel("Players (log)")
pl.ylabel("Estimated Net Revenue (log)")
mu.plot_out()

pl.figure(figsize=figsize, dpi=dpi)
pl.hexbin(data['DaysSinceRelease'],
          np.log10(data['Owners']),
          gridsize=32,
          mincnt=1,
          cmap=pl.cm.winter)
pl.colorbar()          
pl.xlabel("Days since release")
pl.ylabel("Owners (log 10)")
mu.plot_out()

pl.figure(figsize=figsize, dpi=dpi)
pl.hexbin(data['DaysSinceRelease'],
          np.log10(data['Players']),
          gridsize=32,
          mincnt=1,
          cmap=pl.cm.winter)
pl.colorbar()          
pl.xlabel("Days since release")
pl.ylabel("Estimated Players (log 10)")
mu.plot_out()
print("Line artifact is at 500 players")

pl.figure(figsize=figsize, dpi=dpi)
pl.hexbin(data['DaysSinceRelease'],
          np.log(data['Approx Net Revenue']),
          gridsize=32,
          mincnt=1,
          cmap=pl.cm.winter)
pl.colorbar()          
pl.xlabel("Days since release")
pl.ylabel("log Approx Net Revenue")
mu.plot_out()

print(np.sum(data['Median Playtime'] <= 0))
print(np.sum(data['Approx Net Revenue'] <= 0))

pl.figure(figsize=figsize, dpi=dpi)
pl.loglog((data['Median Playtime'].values),
          (data['Approx Net Revenue'].values),
          'bo',
          alpha=0.3, lw=0.2)
pl.axvline(x=(3 * 60 * 60),
           ymin=0, ymax=1, c='r')
pl.annotate('3 hours', xy=((3 * 60 * 60) + 800, 500))
pl.xlabel("Median Playtime (log)")
pl.ylabel("Net Revenue (log)")
mu.plot_out()

pl.figure(figsize=figsize, dpi=dpi)
pl.loglog((data['Avg Playtime']),
          (data['Approx Net Revenue']),
          'bo',
          alpha=0.3, lw=0.1)
pl.axvline(x=(3 * 60 * 60),
           ymin=0, ymax=1, c='r')
pl.annotate('3 hours', xy=((3 * 60 * 60) + 800, 500))
pl.xlabel("Mean Playtime (log)")
pl.ylabel("Net Revenue (log)")
mu.plot_out()

pl.figure(figsize=figsize, dpi=dpi)
pl.scatter(np.log(data['Players']),
           np.log(data['Owners']),
           c = data['DaysSinceRelease'],
           alpha=0.3, lw=0.2)
pl.xlabel("log Players")
pl.ylabel("log Owners")
pl.colorbar()
mu.plot_out()

# Subtrace means for Regression
data['Avg Playtime meaned'] = (data['Avg Playtime'] -
                               np.mean(data['Avg Playtime']))
data['Median Playtime meaned'] = (data['Median Playtime'] -
                                  np.mean(data['Median Playtime']))
data['Players meaned'] = (data['Players'] -
                          np.mean(data['Players']))

if False:
    data['bc Avg Playtime'], _ = boxcox(data['Avg Playtime'])
    data['bc Median Playtime'], _ = boxcox(data['Median Playtime'])
    data['bc Players'], _ = boxcox(data['Players'])
    data['bc DaysSinceRelease'], _ = boxcox(data['DaysSinceRelease'])
    
features = [
    'Avg Playtime meaned',
    'Median Playtime meaned',
    'Players meaned',

    'is_RPG',
#    'is_action',
    'is_puzzle',
    'is_platformer',
    'is_sandbox',
    'is_simulation',
    'is_strategy',
    'is_survival',
    'HasScore',
    'HasScore>50%',
    'DaysSinceRelease',
    'VRInTitle',
    '2InTitle',
    'SpaceInTitle',
    'SuperInTitle',
#    'TheInTitle',
#    'OfInTitle'
]


print("R^2 without feature:")
for f in features:
    new_features = features[:]
    new_features.remove(f)
    
    xs = data[new_features].values.astype(float)
    ys = data['Approx Net Revenue'].values

    m = LinearRegression()
    m.fit(xs, ys)

    score = m.score(xs, ys)

    print("{:20s} {:9.6f}".format(f, score))
print()

xs = data[features].values.astype(float)
ys = data['Approx Net Revenue'].values

if False:
    # Show feature correlation plot
    correlations = np.corrcoef(xs.T)
    pl.imshow(correlations, cmap=pl.cm.jet, interpolation='nearest')
    pl.colorbar()
    mu.plot_out()

m = LinearRegression()
m.fit(xs, ys)

print("R^2: {:8.4f}".format(m.score(xs, ys)))
print()
for feature, coef in zip(features, m.coef_):
    print("{:20s} {:16.4f}".format(feature, coef))
print("{:20s} {:16.4f}".format("intercept", m.intercept_))
print()

if True:
    print(data.ix[0])

print()
vocab = Counter()
letters = Counter()
for game in data['Game']:
    game = game.lower().replace(':', '').replace('-', '').replace('&', '')
    game = game.replace('?', '').replace(')', '').replace('(', '')
    game = game.replace('!', '').replace('"', '').replace("'", '')
    for word in game.lower().split():
        vocab[word] += 1

print(" --- Common words ---")        
for k, v in vocab.most_common(20):
    print("{:20s} {:10d}".format(k, v))

print()    

for i in np.argsort(data['Avg Playtime'])[-20:] :
    print("{:35s} {:9.0f} {:10.0f} {:10.0f}".format(data.iloc[i]['Game'],
                                                data.iloc[i]['Avg Playtime'] / 3600,
                                                data.iloc[i]['Players'],
                                                data.iloc[i]['Owners']))
print()
for i in np.argsort(data['Median Playtime'])[-20:]:
    print("{:35s} {:9.0f} {:10.0f} {:10.0f}".format(data.iloc[i]['Game'],
                                                data.iloc[i]['Median Playtime'] / 3600,
                                                data.iloc[i]['Players'],
                                                data.iloc[i]['Owners']))
print()
for i in np.argsort(data['Players'])[-20:]:
    print("{:35s} {:9.0f} {:10.0f} {:10.0f}".format(data.iloc[i]['Game'],
                                                data.iloc[i]['Avg Playtime'] / 3600,
                                                data.iloc[i]['Players'],
                                                data.iloc[i]['Owners']))
print()
for i in np.argsort(data['Owners'])[-20:]:
    print("{:35s} {:9.0f} {:10.0f} {:10.0f}".format(data.iloc[i]['Game'],
                                                data.iloc[i]['Avg Playtime'] / 3600,
                                                data.iloc[i]['Players'],
                                                data.iloc[i]['Owners']))


