import numpy as np
import pandas as pd

def train_test_split(data):
    train = data.iloc[:int(len(data)*0.8)]
    test = data.iloc[int(len(data)*0.8):]

    #train
    X_train = train.loc[:, train.columns !="diagnosis"]
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    print(X_train.shape)

    Y_train = train["diagnosis"]
    Y_train = np.array(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0], -1).T
    print(Y_train.shape)

    #test
    X_test = test.loc[:, test.columns !="diagnosis"]
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1).T
    print(X_test.shape)

    Y_test = test["diagnosis"]
    Y_test = np.array(Y_test)
    Y_test = Y_test.reshape(Y_test.shape[0], -1).T
    print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test