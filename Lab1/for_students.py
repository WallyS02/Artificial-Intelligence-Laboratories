import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
one = np.ones(len(x_train))
xT = np.vstack((x_train, one))
x = xT.T
theta = np.dot(xT, y_train)
theta_best = np.dot(theta, np.linalg.inv((np.dot(xT, x))))
print("Closed-form solution theta: ", theta_best)

# TODO: calculate error
mse = np.mean((y_train - np.dot(x, theta_best)) ** 2)
print("MSE error: ", mse)

theta_best[0], theta_best[1] = theta_best[1], theta_best[0]

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)
x_std = np.std(x_train)
y_std = np.std(y_train)
x_train = (x_train - x_mean) / x_std
x = np.vstack((x_train, one)).T
y_train = (y_train - y_mean) / y_std

x_test = (x_test - x_mean) / x_std
y_test = (y_test - y_mean) / y_std

# TODO: calculate theta using Batch Gradient Descent
step = 0.001  # learning rate
iterations = 1000
theta_best = np.random.rand(2, 1)
for i in range(0, iterations):
    gradient = -2 * x.T.dot(y_train[:, None] - np.dot(x, theta_best))
    theta_best = theta_best - step * gradient
print("Batch Gradient Descent solution theta: ", theta_best)

# TODO: calculate error
mse_bgd = np.mean((y_train[:, None] - np.dot(x, theta_best)) ** 2)
print("MSE error: ", mse_bgd)

temp1 = theta_best[0]
temp2 = theta_best[1]
theta_best = [temp2, temp1]

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
