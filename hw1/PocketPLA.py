import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ITERATION_TIME = 2000

def sign(num):
    if np.sign(num) <= 0:
        return -1.0
    return 1.0
def pla(X,y,m):
    random.seed()
    flag = True
    pw = np.zeros((1,5))
    random_set = list(range(m))
    random.shuffle(random_set)
    iteration = 0
    while flag:
        flag = False
        for n in random_set:
            w = np.copy(pw + X[n]*y[n])
            total = 0
            correct = 0
            error = 0
            for i in range(m):
                total += pw*X[i].T
                correct += w*X[i].T
                if y[i] != sign(total.sum()):
                    error += 1
                if y[i] != sign(correct.sum()):
                    error -= 1
            if error < 0:
                pw = np.copy(w)
            iteration+=1
            if iteration >= 50:
                break
    return pw
def check(X,y,w):
    m,n = X.shape
    total = 0
    for n in range(m):
        total += w*X[n].T
    return total.sum()

if __name__ == '__main__':

    df = pd.read_csv('./data/hw1_18_train.dat', sep='\s+', index_col=False)
    tmp = df.iloc[:,0:4].values
    (m,n) = tmp.shape
    X = np.ones((m,5))
    X[:,1:] = tmp
    y = df.iloc[:,4].astype(float).tolist()
    df2 = pd.read_csv('./data/hw1_18_test.dat', sep='\s+', index_col=False)
    tmp = df2.iloc[:,0:4].values
    (test_m,n) = tmp.shape
    test_X = np.ones((m,5))
    test_X[:,1:] = tmp
    test_y = df2.iloc[:,4].astype(float).tolist()
    errors = []
    for i in range(ITERATION_TIME):
        w = pla(X,y,m)
        error = check(test_X,test_y,w)
        errors.append(error)

    print(sum(errors)/ITERATION_TIME)
