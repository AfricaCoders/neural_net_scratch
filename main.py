import pandas as pd
import numpy as np
from packages.modules import *

#load data & data cleaning
data = pd.read_csv("data.csv", delimiter=",")
data = data.drop(["Unnamed: 32"], axis= 1)
data["diagnosis"] = data["diagnosis"].replace(["B", "M"],[1, 0])

# split dataset
X_train, Y_train, X_test, Y_test = train_test_split(data)

num_iterations = 15000
np.random.seed(3)
n_x = layer_sizes(X_train, Y_train)[0]
n_h = layer_sizes(X_train, Y_train)[1]
n_y = layer_sizes(X_train, Y_train)[2]

parameters = initialize_parameters(n_x, n_h, n_y)

for i in range(0, num_iterations):
    A2, cache = forward_propagation(X_train, parameters)
    cost = compute_cost(A2, Y_train)
    grads = backward_propagation(parameters, cache, X_train, Y_train)
    parameters = update_parameters(parameters, grads, learning_rate=0.5)

    if i % 1000 == 0:
        print(f"Cost after iteration {i}: {cost}")

train_predictions = predict(parameters, X_train)
test_predictions = predict(parameters, X_test)

print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predictions - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predictions - Y_test)) * 100))
