import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


pth = 'winequality-red.csv'
df = pd.read_csv(pth, sep=";")

X = df['alcohol'].values.reshape(-1, 1)
y = df['quality'].values

X, y, coef = make_regression(n_samples=1000, n_features=1, n_informative=1, shuffle=False, coef=True, noise=1, random_state=42)
'''
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = np.vstack(y)
y = scaler.fit_transform(y)
y = np.hstack(y)
'''

reg = lr().fit(X, y)
y_hat = reg.predict(X)

least_squares = float('inf')
best_slope = best_intercept = 0

slopes = np.arange(0, 20, 0.5)
intercepts = np.arange(-5, 5, 0.1)

ssr_collector = []

for k in slopes:
    print("k: " + str(k) + "\n")
    for d in intercepts:
        y_hat = k * X + d
        y_hat = np.hstack(y_hat)
        sr = (y_hat - y) ** 2
        ssr = np.sum(sr)
        #print("k: " + str(k) + "\nd: " + str(d) + "\n")
        if ssr < least_squares:
            least_squares = ssr
            best_slope = k
            best_intercept = d

y_hat = best_slope * X + best_intercept
tuples = list(zip(y_hat, y))
X = np.hstack(X)
plt.figure(figsize=(20, 15))
sizes = np.full((1, 1599), 4)

plt.plot(X, [i for (i, j) in tuples], 'rx', markersize=7)
plt.plot(X, [j for (i, j) in tuples], 'bo', markersize=7)

plt.xlabel('alcohol content')
plt.ylabel('predicted/true quality')

plt.show()

mse = mean_squared_error(y, y_hat)
