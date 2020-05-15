from flask import Flask, render_template, request, jsonify
import pickle
import time
import numpy as np
import argparse
import re
import itertools

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir

app = Flask(__name__)

data, pred, pred5 = get_data()
env = TradingEnv(data, pred, pred5, 20000)
state_size = env.observation_space.shape
action_size = env.action_space.n
action_combo = list(map(list, itertools.product([0, 1, 2], repeat=3)))
action_map = {0: "sell", 1: "hold", 2: "buy"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def advise():
    n1 = float(request.form['n1'])
    n2 = float(request.form['n2'])
    n3 = float(request.form['n3'])
    cash = float(request.form['cash'])
    print(n1)
    print(cash)

    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)
    agent.load("202005011635-dqn.h5")

    state = env.reset()
    state[0] = n1
    state[1] = n2
    state[2] = n3
    state[-1] = cash
    state = scaler.transform([state])

    action = agent.act(state)
    # action_combo = list(map(list, itertools.product([0, 1, 2], repeat=3)))
    action_vec = action_combo[action]
    # action_map = {0: "sell", 1: "hold", 2: "buy"}

    # print(action_map[action_vec[0]], action_map[action_vec[1]], action_map[action_vec[2]])

    ans = []
    tmp = 1 if action_vec[0] == 0 and n1 == 0 else action_vec[0]
    if cash == 0 and tmp == 2: tmp = 1
    ans.append(action_map[tmp])
    tmp = 1 if action_vec[1] == 0 and n2 == 0 else action_vec[1]
    if cash == 0 and tmp == 2: tmp = 1
    ans.append(action_map[tmp])
    tmp = 1 if action_vec[2] == 0 and n3 == 0 else action_vec[2]
    if cash == 0 and tmp == 2: tmp = 1
    ans.append(action_map[tmp])

    print(ans)
    return render_template('index.html', ans=ans, n1=n1, n2=n2, n3=n3, cash=cash)



if __name__ == '__main__':
    app.run()
