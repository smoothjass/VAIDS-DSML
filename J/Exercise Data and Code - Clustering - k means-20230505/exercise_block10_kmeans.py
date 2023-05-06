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
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


########################################################################################################################
# PART 1 // IMPLEMENT k-Means
########################################################################################################################
'''
In this exercise we want to implement the k-means algorithm
you'll be provided with a set of initial means and you should:
 - (1) assign data points based on euclidean distance to the clusters
 - (2) calculate new centers
    ... interate

data:
-----
- https://www.kaggle.com/aryashah2k/credit-card-customer-data?select=Credit+Card+Customer+Data.csv
- credit card customer data
- we will preselect two features for testing k-means, but be encouraged to try the algorithm on your won

what you don't need to care about:
---------------------------------
- missing values
- categorical values (which are problematic for kmeans anyway)

visualize the cluster updates:
-----------------------------
- show how cluster centers moves over multiple iteration
'''


# load data #
# -- pre-coded -- #
pth = r"ccc.csv"
df = pd.read_csv(pth)
df.describe()
df.dtypes
df.isna().sum()
df.drop(["Sl_No", "Customer Key"], axis=1, inplace=True) # drop keys
df = df.astype("float64") # we will need float values for k-means
df.dtypes

# look at the data (pair plots) #
# select only a few meaningful attributes #
# checkout: https://seaborn.pydata.org/generated/seaborn.jointplot.html #
# -- pre coded -- #
sns.pairplot(df)
plt.show()
# we choose these two features, it seems there are two clusters here
df = df[["Total_visits_online", "Avg_Credit_Limit"]]
sns.jointplot(data=df, x="Total_visits_online", y="Avg_Credit_Limit", kind="kde")
plt.show()
sns.jointplot(data=df, x="Total_visits_online", y="Avg_Credit_Limit", kind="hist")
plt.show()

# implement k-means #
# -- student work -- #
m = np.array([[2.5, 100000],[10.0,125000]]) # we use two cluster centers

# scale the data first and afterwards m ! (So that that m does not jump to extreme values) 

# k-means #1 ... distance and cluster assignment

# k-means #2 ... means update

# visualize results

# iterate







