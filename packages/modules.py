from re import A
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

def sigmoid(z):

    s = 1 / (1 + np.exp(-z))
    return s

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 6
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):


    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1] # number of examples

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = -1/m * np.sum(logprobs)


    return cost

def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis= 1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads

def update_parameters(parameters, grads, learning_rate= 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    for i in range(A2.shape[1]):
        if A2[0, i] > 0.5:
            predictions[0,i] = 1
        else:
            predictions[0, i] = 0

    return predictions