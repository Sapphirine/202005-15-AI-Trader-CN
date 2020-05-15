import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(col='close'):
  """ Returns a 3 x n_step array """
  s1 = pd.read_csv('data/600029.csv', usecols=[col])
  s2 = pd.read_csv('data/600030.csv', usecols=[col])
  s3 = pd.read_csv('data/600031.csv', usecols=[col])
  # recent price are at top; reverse it

  perd = pd.read_csv('data/pred.csv', usecols=[col])
  perd5 = pd.read_csv('data/pred5.csv', usecols=[col])


  return np.array([s1[col].values[::-1],
                   s2[col].values[::-1],
                   s3[col].values[::-1]]), pred, pred5


def get_scaler(env):
  """ Takes a env and returns a scaler for its observation space """
  low = [0] * (env.n_stock * 4 + 1)
  # low = [0] * (env.n_stock * 2 + 1)

  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_cash = env.init_invest * 3 # 3 is a magic number...
  max_stock_owned = max_cash // min_price

  max_pred_price = env.predict_price.max(axis=1)
  max_day_5_pred_price = env.day_5_predict_price.max(axis=1)

  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)

  for i in max_pred_price:
    high.append(i)
  for i in max_day_5_pred_price:
    high.append(i)

  high.append(max_cash)

  scaler = StandardScaler()
  scaler.fit([low, high])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)