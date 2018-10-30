import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

ITERATION_TIME = 1126

def sign(num):
    if np.sign(num) <= 0:
        return -1.0
    return 1.0

def pla(X,y,m):
    random.seed(datetime.now())
    flag = True
    w = np.zeros((1,5), dtype=np.float64)
    random_set = list(range(m))
    random.shuffle(random_set)
    update = 0
    iteration = 0
    while flag:
        flag = False
        for n in random_set:
            total = np.dot(w,X[n].T)
            if y[n] != sign(total.sum()):
                flag = True
                w += X[n]*y[n]
                update += 1
            # iteration+=1
    return update

if __name__ == '__main__':

    df = pd.read_csv('./data/hw1_15_train.dat', sep='\s+', index_col=False)
    tmp = df.iloc[:,0:4].values
    (m,n) = tmp.shape
    X = np.ones((m,5))
    X[:,1:] = tmp
    y = df.iloc[:,4].astype(float).tolist()
    updates = []
    for i in range(ITERATION_TIME):
        update = pla(X,y,m)
        updates.append(update)
        # periods.append(period)
    print(sum(updates)/ITERATION_TIME)
    # print(len(fres), len(updates))
    # bins = np.linspace(0,max(iterations), 100)
    plt.hist(updates,bins='auto', alpha=0.5, label='updates')
    # plt.hist(periods,bins='auto', alpha=0.5, label='periods')
    plt.legend(loc='upper right')
    plt.savefig('./hw1.png')
