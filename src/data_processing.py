import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import pandas as pd
import pickle
import csv
import random

with open("D:\ColumbiaCourses\Advanced Big Data Analytics 6895\data\stock.data", "rb") as f:
    data = pickle.load(f)

keys = data.keys()
stocks = []
ID = []
for i in keys:
    stock = data[i]
    stock = stock.loc[stock.trade_date>'2015000']
    if stock.shape[0] > 1200:
        stocks.append(stock)
        ID.append(i)
#print(len(stocks)) #1225

stocks = stocks[:]
ID = ID[:]
num_stock = len(stocks)

def dateformat(s):
    assert len(s) == 8
    return s[:4]+'-'+s[4:6]+'-'+s[6:8]
df = [pd.DataFrame() for _ in range(num_stock)]
for i in range(num_stock):
    df[i] = stocks[i].loc[:,['trade_date','close','vol']]
    df[i].loc[:,'vol'] = df[i].vol.apply(math.ceil)
    df[i].rename(columns = {'trade_date':'timestamp', 'vol':'volume'},inplace = True)
    df[i].loc[:,'timestamp'] = df[i].timestamp.apply(dateformat)
    

for i in range(num_stock):
    df[i].rename(columns = {'timestamp':'Date', 'close':'Close', 'volume':'Volume'}, inplace = True)

for i in range(num_stock):
    df[i].to_csv('D:\ColumbiaCourses\Advanced Big Data Analytics 6895\milestone3\LSTM-Neural-Network-for-Time-Series-Prediction\data\{}.csv'.format(ID[i]),index=False)


with open('D:\ColumbiaCourses\Advanced Big Data Analytics 6895\milestone3\LSTM-Neural-Network-for-Time-Series-Prediction\data\ID.csv', 'w') as out_f:
    for l in ID:
        out_f.write(l)
        out_f.write('\n')