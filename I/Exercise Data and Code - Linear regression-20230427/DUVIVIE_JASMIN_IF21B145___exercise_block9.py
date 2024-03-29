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
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

########################################################################################################################
# IMPLEMENT AN OLS PARAMETER ESTIMATION FOR LINEAR REGRESSION USING ONLY NUMPY
########################################################################################################################
'''
In this exercise we want you to implement an optimizer for linear regression based on simple linear algebra operation
PLease note: even if this might sound intimidating it is actually very simple!
PLease also note: if you understand the optimizer we use here, you'll be able to solve quite some non-standard
problems without libraries!
AND
In this exercise we want you to visualise and understand linear regression.

Ordinary least squares:
-----------------------
- The optimizer we are talking about is the s.c. ORDINARY LEAST SQUARES optimizer/estimator (OLS for short)
- As the name suggests, OLS minimizes the RSS (residual sum of squares, see also slides p9)
- instead of taking derivatives this problem can also be solved computationally using matrix operations
- this enables us to implement the procedure without caring about manual derivation or using a symbolic math package!
- great, really great it is my friend :)

data:
-----
winequality-red.csv
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
we want to estimate the quality as labelled by "expert-drinkers" based on only ONE feature (alcohol)

what you DONT need to care about:
---------------------------------
- missing data
- categorical/binary features

what you need to care about:
----------------------------
- performing OLS using only numpy!
- the matrix representation of linear regression (multivariate and univariate)

- checkout the following numpy documentation
- https://numpy.org/doc/stable/reference/generated/numpy.matmul.html?highlight=matmul#numpy.matmul
- https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html?highlight=linalg%20inv#numpy.linalg.inv
- https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html

output:
-------
- just comparte the intercept/coefficients found using scikit learn and your own procedure
- plot alcohol content vs the quality predictions with red x-symbols
- plot alcohol content vs the true quality with blue filled circles
- draw a green line between predicted qualities and true qualities to visualise the residuals (= Size of the error we made)
- the result should look like this https://i.stack.imgur.com/zoYKG.png (But with different data!!)
- then label the x-axis and y-axis
- Calculate the Mean Squared Error (MSE)

some help:
----------
- if you are very uncertain look at https://www.fsb.miamioh.edu/lij14/411_note_matrix.pdf
- only slides 3,4,5 and 7 are relevant
- on slide 5 equation (4) is most important, since it shows how to estimate beta (=the coefficients)
'''

def visualize(y, y_hat, X, title):
    # visualise the residuals
    # make tuples from y_hat and y
    tuples = list(zip(y_hat, y))
    # reshape X
    X = np.hstack(X)
    # configure figsize so the plot doesn't seem so crowded
    plt.figure(figsize=(20, 15))

    # plot y (the j in tuple (i, j)
    plt.plot(X, [j for (i, j) in tuples], 'bo', markersize=7)
    # plot y_hat (the i in tuple (i, j)
    plt.plot(X, [i for (i, j) in tuples], 'rx', markersize=7)
    # plot green lines between y_hat and y
    plt.plot((X, X), ([i for (i, j) in tuples], [j for (i, j) in tuples]), c='green')

    # label the axes
    plt.xlabel('alcohol content')
    plt.ylabel('predicted/true quality')
    plt.title(title)
    plt.show()

# read the data --------------------------------------------------
# -- predefined --
pth = 'winequality-red.csv'
df = pd.read_csv(pth, sep=";")
X = df.drop("quality", axis=1).values[:,10:11]
y = df["quality"].values
# generated linear distribution to test with, uncomment to try
# X, y, coef = make_regression(n_samples=1000, n_features=1, n_informative=1, coef=True, noise=1, random_state=42)
_1 = np.repeat(1, X.shape[0]).reshape((X.shape[0], 1)) # we need a vector of 1s to perform OLS using only linear algebra
X1 = np.hstack([_1, X])

# scikit solution ------------------------------------------------
# -- predefined --
# this is just for you to check if your solution correct.
# Using numpy should return the same result (except for rounding)

# fit the line to the data and predict the y values
mod = lr()
mod.fit(X, y)
mod.coef_
mod.intercept_
y_hat = mod.predict(X)

# calculate mean squared error
mse = mean_squared_error(y, y_hat)
visualize(y, y_hat, X, "sklearn")

# preform OLS estimation as described in the doc string ----------
# -- students work --
'''
- the OLS estimation can be performed in 3 steps using only simple/basic linear algebra operations
step by step:
'''

# calculate S1
'''
S1 = inverse(X1'*X1)
- X1 is the matrix X (containing only the features values) with an additional 1-vector prepended in the first column
1-vec| feat1 | feat2
--------------------
 1  | 0.9 | 102
 1  | 1.2 | 908
... | ... | ...

- X1' is the transpose of X1 ... look for matrix transpose in numpy and see what it does
 1  |  1  | ...
0.9 | 1.2 | ...
102 | 908 | ...

- * is matrix multiplication. Please note: this is NOT an element-wise multiplication, if you are uncertain please look
    it up!
- inverse() is the inverse of a matrix ... look how you can invert a matrix in numpy
- info: inverse() is NOT the correct numpy function!
- info: the inverse of a matrix returns the identity matrix when multiplied with the original matrix
- info: This is roughly equivalent to 1/42 being the inverse of 42, but please don't forget that matrix multiplications
    are specially defined functions
'''
X1T = X1.transpose()
S1_temp = np.matmul(X1T, X1)
S1 = np.linalg.inv(S1_temp)
should_be_identity = np.matmul(S1, S1_temp)

# calculate S2
'''
- S2 = X1'*y
- again X1' is the transpose of X1
- * is again matrix multiplication
- info: y is a n*1 matrix, even scalars can be considered to be 1*1 matrices
'''
S2 = np.matmul(X1T, y)

# calculate s3
'''
- S3 = S1*S2
- * is again matrix multiplication
- IMPORTANT: S3 is the result of the OLS estimation
- this numpy array contains
- at index 0 ... the intercept
- at index 1 ... the coefficient of our only feature
- info: the exact same procedure can be used for multivariate regression also!
- for multivariate regressions indices 1 ... n would be the coefficients for our n features
'''
S3 = np.matmul(S1, S2)

# predict and visualize y_hat_man
y_hat_mat = X * S3[1] + S3[0]
y_hat_mat = np.hstack(y_hat_mat)
visualize(y, y_hat_mat, X, "matrix multiplication")

# calculate mean squared error
mse_mat = mean_squared_error(y, y_hat)

# compare results
