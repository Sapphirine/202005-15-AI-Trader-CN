import pickle
import time
import numpy as np
import argparse
import re

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode', type=int, default=2000,
                      help='number of episode to run')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
  parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                      help='initial investment amount')
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
  args = parser.parse_args()

  maybe_make_dir('weights')
  maybe_make_dir('portfolio_val')

  timestamp = time.strftime('%Y%m%d%H%M')

  data, pred, pred5 = get_data()
  train_data = data[:, :3526]
  test_data = data[:, 3526:]

  pred_data_train, pred_data_test = pred[:, :3526], pred[:, 3526:]
  day5_pred_data_train, day5_pred_data_test = pred5[:, :3526], pred5[:, 3526:]

  env = TradingEnv(train_data, pred_data_train, day5_pred_data_train, args.initial_invest)
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  portfolio_value = []
  # actions = []

  # agent.load("weights/202005011635-dqn.h5")

  if args.mode == 'test':
    # remake the env with test data
    env = TradingEnv(test_data, pred_data_test, day5_pred_data_test, args.initial_invest)
    # load trained weights
    agent.load(args.weights)
    # when test, the timestamp is same as time when weights was trained
    timestamp = re.findall(r'\d{12}', args.weights)[0]

  for e in range(args.episode):
  # for e in range(1):
    state = env.reset()
    state = scaler.transform([state])
    for time in range(env.n_step):
      action = agent.act(state)
      # actions.append(action)
      next_state, reward, done, info = env.step(action)
      next_state = scaler.transform([next_state])
      if args.mode == 'train':
        agent.remember(state, action, reward, next_state, done)
      state = next_state
      if done:
        print("episode: {}/{}, episode end value: {}".format(
          e + 1, args.episode, info['cur_val']))
        portfolio_value.append(info['cur_val']) # append episode end portfolio value
        break
      if args.mode == 'train' and len(agent.memory) > args.batch_size:
        agent.replay(args.batch_size)
    if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
      agent.save('weights/{}-dqn.h5'.format(timestamp))

  # save portfolio value history to disk
  with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
    pickle.dump(portfolio_value, fp)
  # with open('actions.p', 'wb') as fp:
  #   pickle.dump(actions, fp)