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
import os

########################################################################################################################
# PART 1 // IMPLEMENT kNN
########################################################################################################################
'''
- implement a simple knn function using only basic python/numpy/pandas
- the pre-coded section will provide a dataset for "training" (a sample from the banknote dataset)
- the pre-coded section will also provide some test points which you should use for prediction
- your function should be able to 
    - provide a prediction y_hat (classification) based on the manhatten distance
    - it should also return the mean manhatten distance of the neighbors for k = 5
    - it should alo return a list, converted to a string, that indicates the row indices of the found neighbors
    - implement k as a parameter

    Data:
    -----
        - we will use the banknote authentication data https://archive.ics.uci.edu/ml/datasets/banknote+authentication
        - 1. variance of Wavelet Transformed image (continuous)
        - 2. skewness of Wavelet Transformed image (continuous)
        - 3. curtosis of Wavelet Transformed image (continuous)
        - 4. entropy of image (continuous)
        - 5. class (integer)

    When implementing the function, you DONT need to care about:
    ------------------------------------------------------------
        - missing values
        - scaling
        - being able to set another distance measure
        - alerting for unclear classification (e.g. for when the top 2 classes in a 3-class problem occur equally often in the kNNs)
        - weighting kNNs by distance

    When implementing the function, SHOULD care about:
    --------------------------------------------------
        - k should be a user parameter
        - your function should return a prediction (y_hat, classification)
        - your function should return the mean manhatten distance of all k neighbors
        - your function should return the row-indices of the found neighbours (casted to a string) 
            - you might find this non-sensical, but in real, tricky applications one might be really interested 
              in what your prediction is actually based on
            - when getting weird results, one reason might be that the neighbors contain dirty data points

    Your output should be a pandas df with the following characteristics:
    ---------------------------------------------------------------------
        - columns
            - yhat        int64 (predicted class label)
            - mndist    float64 (mean manhatten distance based on the kNNs found)
            - idx        object (list of integers --> CASTED INTO A STRING(!!), row index of the kNNs in the input dataframe)
    '''


def knn_classifier(X_train, y_train, X_test, y_test, k):
    output = pd.DataFrame(columns=['yhat', 'mndist', 'idx'])

    for j in range(len(X_test)):
        neighbor_collector = []
        manhattan_dist = abs(X_test[j][0] - X_train[0][0]) + \
                         abs(X_test[j][1] - X_train[0][1]) + \
                         abs(X_test[j][2] - X_train[0][2]) + \
                         abs(X_test[j][3] - X_train[0][3])
        neighbor_collector.insert(0, [0, manhattan_dist])

        for i in range(1, len(X_train)):
            manhattan_dist = abs(X_test[j][0] - X_train[i][0]) + \
                             abs(X_test[j][1] - X_train[i][1]) + \
                             abs(X_test[j][2] - X_train[i][2]) + \
                             abs(X_test[j][3] - X_train[i][3])
            if manhattan_dist < neighbor_collector[0][1]:
                neighbor_collector.insert(0, [i, manhattan_dist])

        class_zero = class_one = mndist = 0
        idx_collection = []

        for n in range(5):
            mndist = mndist + neighbor_collector[n][1]
            idx_collection.append(neighbor_collector[n][0])

            if y_train[neighbor_collector[n][0]] == 0:
                class_zero += 1
            else:
                class_one += 1
        prediciton = 1 if class_zero < class_one else 0
        data_row = pd.Series({'yhat': prediciton, 'mndist': (mndist / 5), 'idx': ' '.join(str(e) for e in idx_collection)})
        print(data_row)
        pd.concat([output, data_row.to_frame().T], ignore_index=True)
    return output


# read data --------------------------------------------
# -- precoded --
pth = 'data_part1_banknote.txt'
cols = ["wavelet_var", "skew_wavelet", "curtos_wavelet", "entropy_img", "class"]
df = pd.read_csv(pth, sep=',', header=None, names=cols)

# sample from original DataFrame
np.random.seed(42)
n = min(500, df["class"].value_counts().min())
df = df.groupby("class").apply(lambda x: x.sample(n))
df.index = df.index.droplevel(0)
df.reset_index(inplace=True, drop=True)
df = df.round(3)
df["class"] = df["class"].astype("int64")

# test points
tps = np.array([[0.100, 5.512, -0.327, 1.001],
                [5.5, 11.987, 13.2, 1.99],
                [4.98, 10.21, 1.76, -0.5],
                [0.5, 5.43, -5.001, -8.0],
                [0.78, 1.61, 2.345, 1.32]])

# select X,y
X_sample = df.iloc[:, 0:4].values
y_sample = df.iloc[:, 4].values
result = knn_classifier(X_sample, y_sample, tps, [], 5)

# knn ---------------------------------------------------------------------
# -- student work --
