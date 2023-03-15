import math

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


'''
- in this example we perform monte carlo cross validation

dataset info:
-------------
    - https://www.kaggle.com/mssmartypants/water-quality

what to do:
-----------

    --> loop over k-settings 1,2,3,...7
    --> for each k-setting use a StratifiedShuffleSplit(), test_size=0.2, n_splits=5, random_state=42 for performance evaluation
        - use the following metrics:
            - accuracy()
            - balanced_accuracy()
            - brier_score_loss()
                - hint: this metrics works similar to roc_auc_score()
                - you need to use predict_proba()
                - the you have to select probabilities for the positive label (1 in this case)
                - the you can use y_test and y_hat_proba for class 1 only to get the breir score
            - ROUND ALL RESULTS TO 3 DECIMAL PLACES!

        - collect the results (the means of accuracies for each fold) for each k-setting in a dictionary
            - if you are unsure look at ex2(), this is the same logic but for classification
            - {"k":_, "accuracy":_, "balanced_accuracy":, "brier_score":_}

        - collect the dictionary for each k setting in a list, s.t. it can be easily transformed into a DataFrame

    --> transform the collected list of dictionaries into a pandas DataFrame
    --> your result should look like:
        k         int64
        acc     float64
        bacc    float64
        bsc     float64

    --> your result should have shape (7,4)

questions to think about:
-------------------------
    - whats the difference (numerically) between ass/bacc and bsc?
    - is there any difference in interpretation?
'''

# read the data ----------------------
# -- pre coded --
pth = 'data_ex1ex3_waterquality.csv'
df = pd.read_csv(pth, sep=";")

# explore a little -------------------
# classes are unbalanced!
# -- pre coded --
df[["is_safe"]].groupby(["is_safe"]).size()

# extract X,y arrays -----------------
# -- student work --
#X = ...
#y = ...
y = df['is_safe'].to_numpy()
df.drop(['is_safe'], inplace=True, axis=1)
X = df.to_numpy()

# scale data -------------------------
# -- pre coded --
mmSc = MinMaxScaler()
X = mmSc.fit_transform(X)

# monte carlo cross validation -------
# use k-settings 1,2,3,...7
# -- student work --
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
collector = []

for k in np.arange(1,8,1):

    knn = KNeighborsClassifier(k)
    collector_acc = []
    collector_bacc = []
    collector_bsc = []

    for train_idx, test_idx in sss.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        knn.fit(X_train, y_train)
        y_hat = knn.predict(X_test)
        y_hat_proba = knn.predict_proba(X_test)

        collector_acc.append(accuracy_score(y_test, y_hat))
        collector_bacc.append(balanced_accuracy_score(y_test, y_hat))
        collector_bsc.append(brier_score_loss(y_test, y_hat_proba[:,1]))

    collector.append({"k": k,
                      "acc": np.round(np.mean(collector_acc), 3),
                      "bacc": np.round(np.mean(collector_bacc), 3),
                      "bsc": np.round(np.mean(collector_bsc), 3)})

# build and return the DataFrame
# -- student work --

new_df3 = pd.DataFrame(collector, columns=['k', 'acc', 'bacc', 'bsc'])
