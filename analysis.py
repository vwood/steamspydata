#!/usr/bin/env python3

import numpy as np
from zipfile import ZipFile
import pandas as pd
import scipy as sp
from sklearn import preprocessing
from sklearn import decomposition
import myutils as mu
import pylab as pl
import glob

# print(glob.glob("data/*"))

d1 = pd.read_csv('data/2016 Games on Steam - Indie AND Action.csv')
d1.rename(columns={'c': 'c_indie&action', '#': '#_indie&action'}, inplace=True)
d1['is_action'] = 1
d1['is_indie'] = 1

d2 = pd.read_csv('data/2016 Games on Steam - Indie AND Strategy.csv')
d2.rename(columns={'c': 'c_indie&strategy', '#': '#_indie&strategy'}, inplace=True)
d2['is_strategy'] = 1
d2['is_indie'] = 1

d3 = pd.read_csv('data/2016 Games on Steam - Indie AND Sandbox.csv')
d3.rename(columns={'c': 'c_indie&sandbox', '#': '#_indie&sandbox'}, inplace=True)
d3['is_sandbox'] = 1
d3['is_indie'] = 1

d4 = pd.read_csv('data/2016 Games on Steam - Indie.csv')
d4.rename(columns={'c': 'c_indie', '#': '#_indie'}, inplace=True)
d4['is_indie'] = 1

d5 = pd.read_csv('data/2016 Games on Steam - Indie AND Survival.csv')
d5.rename(columns={'c': 'c_indie&survival', '#': '#_indie&survival'}, inplace=True)
d5['is_survival'] = 1
d5['is_indie'] = 1

d6 = pd.read_csv('data/2016 Games on Steam - Indie AND Simulation.csv')
d6.rename(columns={'c': 'c_indie&simulation', '#': '#_indie&simulation'}, inplace=True)
d6['is_simulation'] = 1
d6['is_indie'] = 1

d7 = pd.read_csv('data/2016 Games on Steam - Platformer AND Puzzle.csv')
d7.rename(columns={'c': 'c_platformer&puzzle', '#': '#_platformer&puzzle'}, inplace=True)
d7['is_platformer'] = 1
d7['is_puzzle'] = 1

d8 = pd.read_csv('data/2016 Games on Steam - Indie AND Platformer.csv')
d8.rename(columns={'c': 'c_indie&platformer', '#': '#_indie&platformer'}, inplace=True)
d8['is_platformer'] = 1
d8['is_indie'] = 1

d9 = pd.read_csv('data/2016 Games on Steam - Indie AND RPG.csv')
d9.rename(columns={'c': 'c_indie&RPG', '#': '#_indie&RPG'}, inplace=True)
d9['is_RPG'] = 1
d9['is_indie'] = 1

data = pd.DataFrame()
data['Game'] = pd.Series(list(set(d1['Game'])
                              .union(set(d2['Game']))
                              .union(set(d3['Game']))
                              .union(set(d4['Game']))
                              .union(set(d5['Game']))
                              .union(set(d6['Game']))
                              .union(set(d7['Game']))
                              .union(set(d8['Game']))
                              .union(set(d9['Game']))))

data = data.merge(d1, on='Game', how='left', sort=True)
data['is_strategy'] = np.nan
data['is_sandbox'] = np.nan
data['is_survival'] = np.nan
data['is_simulation'] = np.nan
data['is_platformer'] = np.nan
data['is_puzzle'] = np.nan
data['is_RPG'] = np.nan

print(data.head(20))
data = data.merge(d2, on='Game', how='left', sort=True)
# data.combine_first(d2)
print()
print(data.head(20))

exit()

        

#data['Strategy'] = 1



print(data.head())
# data['Game', 'Release date', 'Price', 
# print(data.head())

exit()

