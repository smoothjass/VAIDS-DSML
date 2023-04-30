########################################################################################################################
# REMARKS
########################################################################################################################

'''
## Coding
- please note, this no SE course and much of the code in ML is more akin to executing workflows
- please try to use the scripts as a documentation of your analysis, including comments, results and interpretations

## GRADING
- Please refer to the moodle course for grading information

## UPLOAD
- upload your solution on Moodle as: "yourLASTNAME_yourFIRSTNAME_yourMOODLE-UID___exercise_blockX.py"
- please no non-ascii characters on last/first name :)
- NO zipfile, NO data!, ONLY the .py file!

## PRE-CODED Parts
- all exercises might contain parts which where not yet discussed in the course
- these sections are pre-coded then and you are encouraged to research their meaning ahead of the course
- so, if you find something in the exercise description you haven´t heard about, look at the ode section and check if this is pre-coded

## ERRORS
- when you open exercise files you'll see error (e.g. unassigned variables)
- this is due to the fact that some parts are missing, and you should fill them in
'''

########################################################################################################################
# IMPORTS
########################################################################################################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

########################################################################################################################
# IMPLEMENT AN OLS PARAMETER ESTIMATION FOR LINEAR REGRESSION USING ONLY NUMPY
########################################################################################################################

'''
In this exercise we want you to visualise and understand linear regression.

data:
-----
winequality-red.csv
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
we want to estimate the quality as labelled by "expert-drinkers" based on only ONE feature (alcohol)

what you DONT need to care about:
---------------------------------
- missing data
- categorical/binary features

output:
-------
- use sklearn.linear_model to predict quality based on alcohol content
- plot alcohol content vs the quality predictions with red x-symbols
- plot alcohol content vs the true quality with blue filled circles
- draw a green line between predicted qualities and true qualities to visualise the residuals (= Size of the error we made)
- the result should look like this https://i.stack.imgur.com/zoYKG.png (But with different data!!)
- then label the x-axis and y-axis
- Calculate the Mean Squared Error (MSE)
'''

# read the data --------------------------------------------------
# -- predefined --
pth = 'winequality-red.csv'
df = pd.read_csv(pth, sep=";")

X = df['alcohol'].values.reshape(-1, 1)
y = df['quality'].values

# generated linear distribution to test with, uncomment to try
# X, y, coef = make_regression(n_samples=1000, n_features=1, n_informative=1, coef=True, noise=1, random_state=42)

'''
# i've tried scaling out of curiosity but it doesn't change a thing. maybe it would for multi-linear regression? i don't
# think so though
# print(np.unique(y))
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = np.vstack(y)
y = scaler.fit_transform(y)
y = np.hstack(y)
'''

# solution
# fit the line to the data and predict the y values
reg = lr().fit(X, y)
y_hat = reg.predict(X)

# calculate mean squared error
mse = mean_squared_error(y, y_hat)

# visualise the residuals
# make tuples from y_hat and y
tuples = list(zip(y_hat, y))
# reshape X
X = np.hstack(X)
# configure figsize so the plot doesn't seem so crowded
plt.figure(figsize=(20, 15))

# plot y (the j in tuple (i, j)
plt.plot(X, [j for (i, j) in tuples], 'bo', markersize = 7)
# plot y_hat (the i in tuple (i, j)
plt.plot(X, [i for (i, j) in tuples], 'rx', markersize = 7)
# plot green lines between y_hat and y
plt.plot((X, X), ([i for (i, j) in tuples], [j for (i, j) in tuples]), c='green')

# label the axes
plt.xlabel('alcohol content')
plt.ylabel('predicted/true quality')

plt.show()

########################################################################################################################
# TRIED TO FIT THE MODEL MANUALLY
########################################################################################################################

# set least_squares to infinity, so I can start comparing later
least_squares = float('inf')
# initialize best slope/intercept with any value, doesn't matter what
best_slope = best_intercept = 0

# make iterable arrays for slopes and intercepts to try (note accuracy/speed trade-off)
# slower but more accurate
slopes = np.arange(0, 20, 0.1)
intercepts = np.arange(-2, 2, 0.01)

# faster but less accurate, comment next two lines to make it slower but more accurate
slopes = np.arange(0, 20, 0.5)
intercepts = np.arange(-5, 5, 0.1)

# try combinations of slopes and intercepts
for k in slopes:
    print("k: " + str(k) + "\n")
    for d in intercepts:
        y_hat_man = k * X + d
        # y_hat needs to be hstacked so that the sr calculation works correctly
        y_hat_man = np.hstack(y_hat_man)
        # calculate squared residuals
        sr = (y_hat_man - y) ** 2
        # calculate sum of squared residuals
        ssr = np.sum(sr)
        # if the new sum of squared residuals is smaller than the currently smallest value of least_squares, replace the
        # least_squares with the new ssr and the currently best slope/intercept with the new slope/intercept
        if ssr < least_squares:
            least_squares = ssr
            best_slope = k
            best_intercept = d

# predict y_hat with the slope and intercept that came out of the loop as the best fits
y_hat_man = best_slope * X + best_intercept
# calculate mean squared error
mse_man = mean_squared_error(y, y_hat_man)

# same procedure as above
tuples = list(zip(y_hat_man, y))
plt.figure(figsize=(20, 15))

plt.plot(X, [j for (i, j) in tuples], 'bo', markersize=7)
plt.plot(X, [i for (i, j) in tuples], 'rx', markersize=7)

plt.xlabel('alcohol content')
plt.ylabel('predicted/true quality')

plt.show()
