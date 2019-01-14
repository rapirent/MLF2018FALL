import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

ETA = 0.01
ETA2 = 0.001

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sign(num):
    if num <= 0:
        return -1.0
    return 1.0

def error_cal(X,y,m,w):
    error = 0.0
    for n in range(m):
        if(sign(np.dot(w,X[n]).sum())!=y[n]):
            error += 1
    return error/m

def logistic(X,y,m,w):
    gradient = np.zeros((1,20), dtype=np.float64)
    for n in range(m):
        gradient += sigmoid((-1)*y[n]*np.dot(w,X[n]).sum()) * (-1) * y[n] * X[n]
    gradient /= m
    # wt+1 = wt −η∇Ein(wt)
    w = w - (ETA * gradient)
    return w

def sgd(X,y,m,w,n):
    # n = random.choice(range(m))
    gradient = sigmoid((-1)*y[n%m]*np.dot(w,X[n%m]).sum()) * (-1) * y[n%m] * X[n%m]
    w = w - (ETA2 * gradient)
    return w

if __name__ == '__main__':
    random.seed(datetime.now())
    df = pd.read_csv('./data/hw3_train.dat', sep='\s+', index_col=False)
    X = df.iloc[:,0:20].values
    (m,n) = X.shape
    y = df.iloc[:,20].astype(float).tolist()
    e_in_l = []
    e_in_s = []
    e_out_l = []
    e_out_s = []
    w = np.zeros((1,20), dtype=np.float64)
    w2 = np.zeros((1,20), dtype=np.float64)
    df_test = pd.read_csv('./data/hw3_test.dat', sep='\s+', index_col=False)
    X_test = df_test.iloc[:,0:20].values
    (m_test,n) = X_test.shape
    y_test = df_test.iloc[:,20].astype(float).tolist()
    for n in range(2000):
        w = logistic(X,y,m,w)
        w2 = sgd(X,y,m,w2,n)
        e_in_l.append(error_cal(X,y,m,w))
        e_in_s.append(error_cal(X,y,m,w2))
        e_out_l.append(error_cal(X_test,y_test,m_test,w))
        e_out_s.append(error_cal(X_test,y_test,m_test,w2))
    print('gd Ein',error_cal(X,y,m,w))
    print('gd Eout',error_cal(X_test,y_test,m_test,w))
    print('sgd Ein',error_cal(X,y,m,w2))
    print('sgd Eout',error_cal(X_test,y_test,m_test,w2))
    plt.plot(e_in_l,color='olive', alpha=0.5, label='gd')
    plt.plot(e_in_s,color='green', alpha=0.5, label='sgd')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('E_in')
    plt.title('hw2-q4')
    plt.savefig('./hw3-q4.png')
    plt.clf()
    plt.plot(e_out_l,color='olive', alpha=0.5, label='logistic regression')
    plt.plot(e_out_s,color='green', alpha=0.5, label='sgd')
    plt.legend(loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('E_out')
    plt.title('hw2-q5')
    plt.savefig('./hw3-q5.png')
    plt.clf()



