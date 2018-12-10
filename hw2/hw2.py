import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

DATASIZE = 20
ITERATION_TIME = 1000
# x = np.random.uniform(0, 1, DATASIZE)

def sign(num):
    if num <= 0:
        return -1.0
    return 1.0

def generate_data():
    random.seed(datetime.now())
    x = np.random.uniform(-1, 1, DATASIZE)
    x.sort()
    y = []
    weighted_random = [-1] * 20 + [1] * 80
    for index, _x in enumerate(x):
        # noise = np.random.choice([1,-1],1,[0.8,0.2])
        noise = random.choice(weighted_random)
        y.append(sign(_x)*noise)
    return x, y

def decision_stump(x,y):
    theta = []
    x.sort()
    for index in range(DATASIZE):
        # median of the range
        if (index - 1 < 0 ):
            theta.append((x[index] - 1)/2)
        else:
            theta.append((x[index] + x[index-1])/2)
    theta.append((x[-1] + 1)/2)
    e_in_best = 1000000
    for _theta in theta:
        error_neg = 0
        error_pos = 0
        for index, _x in enumerate(x):
            h = -1 * sign( _x - _theta)
            if (h != y[index]):
                error_neg += 1

            h = sign(_x - _theta)
            if (h != y[index]):
                error_pos += 1

        if (error_pos < error_neg):
            better_s = 1
            e_in = error_pos / DATASIZE
        else:
            better_s = -1
            e_in = error_neg / DATASIZE

        if (e_in < e_in_best):
            hypothesis_best = (better_s, _theta)
            e_in_best = e_in
    # print(hypothesis_best, hypothesis_best[0], hypothesis_best[1])
    e_out = 0.5 + 0.3*hypothesis_best[0]*(abs(hypothesis_best[1]) - 1)

    return e_out, e_in_best

if __name__ == "__main__":
    estimate = []
    e_ins = []
    e_outs = []
    for _ in range(ITERATION_TIME):
        x, y = generate_data()
        e_out, e_in = decision_stump(x, y)
        e_ins.append(e_in)
        e_outs.append(e_out)
        estimate.append(e_in - e_out)

    print('avergage e_in =  {}'.format(sum(e_ins)/ITERATION_TIME))
    print('avergage e_out =  {}'.format(sum(e_outs)/ITERATION_TIME))

    plt.hist(estimate,bins='auto', alpha=0.5, label='h_in - h_out')
    # plt.hist(periods,bins='auto', alpha=0.5, label='periods')
    plt.legend()
    plt.savefig('./hw2.png')
