########################################################################################################################
# REMARKS
########################################################################################################################
'''
## Coding
- please note, this not SE course and much of the code in ML is more akin to executing workflows
- please try to use the scripts as a documentation of yur analysis, including comments, results and interporetations

## GRADING
- Please refer to the moodle course for grading information

## UPLOAD
- upload your solution on Moodle as: "yourLASTNAME_yourFIRSTNAME_yourMOODLE-UID___exercise_blockX.py" </b>
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
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


########################################################################################################################
# PART 1 // TRAIN/TEST SPLITS
########################################################################################################################


'''
in this exercise you should perform performance evaluation using a train-test split
this exercise will also demonstrate issues with unbalanced classes, differences between balanced and unbalanced accuracy scores
also, this exercise will demonstrate that the ordering of your data and how you handle it is important!
if you have trouble please check the demo code belonging to this section

dataset info:
-------------
    - https://www.kaggle.com/mssmartypants/water-quality

what to do:
-----------
    --> read the data
    --> check class balancing
    --> extract X and y arrays - you dont need to care about outliers etc
    --> since we use knn again, scale the data

    --> performance evaluation using a simple train_test_split(), test_size=0.2, shuffle=False
        - use the obtained train/test split to evaluate:
            - accuracy()
            - balanced_accuracy()
            - roc_auc_score()
            - for k settings of 1,2,3,4,5,6,7
                hint: define train/test splits first, then loop over the different k-settings
            - ROUND ALL RESULTS TO 3 DECIMAL PLACES!

        - collect the results for each k-setting in a dictionary
            - {"k":_, "accuracy":_, "balanced_accuracy":, "roc_auc":_, "shuffle":False}

        - collect the dictionary for each k setting in a list, s.t. it can be easily transformed into a DataFrame

    --> perform evaluation using a simple train_test_split(), test_size=0.2, BUT THIS TIME SET shuffle=True! and random_state=42
        - all other details are as above (e.g. collect different performance measures etc..)

    --> after performance evaluation with shuffle=False/True you should have:
        - two lists of length 7, each element is a dictionary with values for "k", "accuracy", "balanced_accuracy", "roc_auc" and "shuffle"
        --> put these two lists together s.t. results with shuffle=False come first
        --> transform into a pandas DataFrame

    --> your result should look like:
        k                      int64
        accuracy             float64
        balanced_accuracy    float64
        roc_auc              float64
        shuffle                 bool

    --> your result should have shape (14,5)

questions to think about:
-------------------------
    - where does the difference between accuracy and balanced_accuracy come from?
    - what is the better metric in this case?
    - do you need to care for balancing when using the ROC-AUC?

'''

# read the data ----------------------
# -- pre coded --
pth = 'data_ex1ex3_waterquality.csv'
df = pd.read_csv(pth, sep=";")

# explore a little -------------------
# classes are unbalanced! (attribute is_safe = y)
# -- pre coded --
df[["is_safe"]].groupby(["is_safe"]).size()

# extract X,y arrays -----------------
# -- student work --


# scale data -------------------------
# -- pre coded --
mmSc = MinMaxScaler()
X_scale = mmSc.fit_transform(X)

# perform holdout validation without shuffle and stratify -------
# use k settings 1,2,3,...7
# -- student work --

# perform holdout validation WITH shuffle but NO stratify -----
# again, use k settings 1,2,3,...7
# -- student work --

# build the dataframe
# -- student work --


########################################################################################################################
# PART 2 X-VAL
########################################################################################################################


'''
in this exercise we want to evaluate the performance of a regression model

dataset info:
-------------
    - https://www.kaggle.com/kukuroo3/mosquito-indicator-in-seoul-korea

what to do:
-----------
    --> read and preprocess the data
    --> extract X and y
        - y = column "mosquito_Indicator"
    --> scale the data

    --> performance evaluation: loop over all possible k setting from 1,2,3,... 7
        - for each setting of k perform X-validatation using KFold()
            - set n_splits to 5 and shuffle to True, random_state to 42
            - use the following metrics
                - mean_absolute_error()
                - the root mean spared error (square root of mean_squared_error())
                - median_absolute_error()

        - for each setting of k calculate the mean of the above mentioned regressionaccuracy metrics
        - for each setting of k, collect results in a dictionary
            - {"k":_, "mae":_, "rmse":_, "medae":_}
        - as in ex1() build a dataframe from all collected dictionaries

    --> your solution should look like this:
        - k          int64
        - mae      float64
        - rmse     float64
        - medae    float64

    --> your solution should have shape (7,4)

questions to think about:
-------------------------
    - what do calculated error values mean in practice?
    - How could you say if this is good enough or not?
    - What is a resiudal plot and how could you use it to assess model performance?
'''

# read the data ----------------------
# -- precoded --
df = pd.read_csv(pth, sep=",")
df[["year", "month", "day"]] = df["date"].str.split("-", expand=True)
df[["year", "month"]] = df[["year", "month"]].astype(int)
df.drop(["date", "day"], axis=1, inplace=True)

# extract X,y arrays -----------------
# -- student work --
# X = ...
# y = ...

# scale data -------------------------
# -- precoded --
mmSc = MinMaxScaler()
X_scale = mmSc.fit_transform(X)

# perform cross validation with 5 folds ----
# -- students work --
collector = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for k in np.arange(1, 8, 1):

    knn = KNeighborsRegressor(k)
    collector_mae = []
    collector_rmse = []
    collector_medae = []

    for train_idx, test_idx in kf.split(X_scale, y):

    # ...
    # ...
    # ...

    collector.append({"k": k,
                      "mae": np.mean(collector_mae),
                      "rmse": np.mean(collector_rmse),
                      "medae": np.mean(collector_medae)})

# build DataFrame and return ----------
# -- students work --


########################################################################################################################
# PART 3 // Monte Carlo Validation
########################################################################################################################


'''
- in this example we perform monte carlo cross validation

dataset info:
-------------
    - https://www.kaggle.com/mssmartypants/water-quality

what to do:
-----------
    --> read the data
    --> check class balancing
    --> extract X and y arrays - you dont need to care about outliers etc
    --> since we use knn again, scale the data

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
df = pd.read_csv(pth, sep=";")

# explore a little -------------------
# classes are unbalanced!
# -- pre coded --
df[["is_safe"]].groupby(["is_safe"]).size()

# extract X,y arrays -----------------
# -- student work --
#X = ...
#y = ...

# scale data -------------------------
# -- pre coded --
mmSc = MinMaxScaler()
X_scale = mmSc.fit_transform(X)

# monte carlo cross validation -------
# use k-settings 1,2,3,...7
# -- student work --

# ...
# ...
# ...

for k in np.arange(1,8,1):

    knn = KNeighborsClassifier(k)
    collector_acc = []
    collector_bacc = []
    collector_bsc = []

    for train_idx, test_idx in # ...

# ...
# ...

# build and return the DataFrame
# -- student work --



