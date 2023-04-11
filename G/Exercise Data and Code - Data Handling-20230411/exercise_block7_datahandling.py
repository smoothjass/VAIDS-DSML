########################################################################################################################
# EX1
########################################################################################################################


'''
load dataset "wine_exercise.csv" and try to import it correctly using pandas/numpy/...
the dataset is based on the wine data with some more or less meaningful categorical variables
the dataset includes all kinds of errors
    - missing values with different encodings (-999, 0, np.nan, ...)
    - typos for categorical/object column
    - columns with wrong data types
    - wrong/mixed separators and decimals in one row!
        - please note, this is a very unpleasant error!
    - "slipped values" where one separator has been forgotten and values from adjacent columns land in one column
    - combined columns as one column
    - unnecessary text at the start/end of the file
    - ...

(1) repair the dataset
    - consistent NA encodings. please note, na encodings might not be obvious at first ...
    - correct data types for all columns
    - correct categories (unique values) for object type columns
    - read all rows, including those with wrong/mixed decimal, separating characters

(2) find duplicates and exclude them
    - remove only the unnecessary rows

(3) find outliers and exclude them - write a function to plot histograms/densities etc. so you can explore a dataset quickly
    - just recode them to NA
    - proline (check the zero values), magnesium, total_phenols
    - for magnesium and total_phenols fit a normal and use p < 0.025 as a cutoff value for identifying outliers
    - you should find 2 (magnesium) and  5 (total_phenols) outliers

(4) impute missing values using the KNNImputer
    - including the excluded outliers! (impute these values)
    - use only the original wine features as predictors! (no age, season, color, ...)
    - you can find the original wine features using load_wine()
    - never use the target for imputation!

(5) find the class distribution
    - use the groupby() method

(6) group magnesium by color and calculate statistics within groups
    - use the groupby() method
'''
import numpy as np

########################################################################################################################
# Solution
########################################################################################################################


'''
PLease note:
- the structure below can help you, but you can also completely ignore it
'''

# set pandas options to make sure you see all info when printing dfs
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# ----------------------------------------------------------------------------------------------------------------------
# >>> step 1, just try it
# ----------------------------------------------------------------------------------------------------------------------
'''
start by just loading the data
'''
df = pd.read_csv('wine_exercise.csv', sep=';')

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 2, use skip rows
# ----------------------------------------------------------------------------------------------------------------------
'''
skip row to ignore text inside the file
'''
df = pd.read_csv('wine_exercise.csv', sep=';', skiprows=1)

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 3, use skipfooter
# ----------------------------------------------------------------------------------------------------------------------
'''
the footer is also a problem, skip it with skipfooter
'''
df = pd.read_csv('wine_exercise.csv', sep=';', skiprows=1, skipfooter=1, engine='python')
df.info()
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 4, check na, data types
# ----------------------------------------------------------------------------------------------------------------------
'''
now the df looks fine, but is it really?
only 3 attributes should be categorical but many are, invesigate those
you'll need to set pandas options to see all issues
'''
'''
- i guess by 'categorical' they don't mean the actual dtype category because actually none of these attributes is 
displayed as a category with df.info() but many are object when they should be float or something else
- i have read up on some options but i don't understand how i should set them, which ones i should set, i'm confused
'''

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 5 try to convert data types to find issues
# ----------------------------------------------------------------------------------------------------------------------
'''
hint: rows 50, 51, 142 are problematic due to mixed/wrong separators or wrong commas
How could you find such issues in an automated way?
'''
#df = pd.read_csv('wine_exercise.csv', sep=';', skiprows=1, skipfooter=1, engine='python')
#df.astype({'alcohol': 'float64'}).dtypes

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 6, exclude the three problematic rows
# ----------------------------------------------------------------------------------------------------------------------
'''
the three rows are completely ruined and can only be fixed in isolation
you can read the dataset an skip these rows
'''
#and also 166 because there is a white space instead of a ';' as a separator
df = pd.read_csv('wine_exercise.csv', sep=';', decimal='.', skiprows=[0, 52, 53, 144, 168], skipfooter=1, engine='python',
                 na_values='missing')
df.astype({'alcohol': 'float64', 'ash': 'float64'}).dtypes
df.info()

# ----------------------------------------------------------------------------------------------------------------------
# step 7, handle rows separately
# ----------------------------------------------------------------------------------------------------------------------
'''
If this is too much data dirt for you continue without handling these three rows (continue with step 8)
Otherwise you can follow the workflow indicated below (steps 7.1, 7.2, 7.3, 7.4)
'''

# ----------------------------------------------------------------------------------------------------------------------
# step 7.1, first get the column names from the df
'''
get column names so you can assign them to the single rows you did read
'''

# ----------------------------------------------------------------------------------------------------------------------
# step 7.2, handle row 52
'''
read only row 52 and repair it in isolation
write it to disk wit correct separators, decimals
'''


# ----------------------------------------------------------------------------------------------------------------------
# step 7.3, handle row 53
'''
read only row 53 and repair it in isolation
write it to disk wit correct separators, decimals
'''

# ----------------------------------------------------------------------------------------------------------------------
# step 7.4, handle row 144
'''
read only row 144 and repair it in isolation
write it to disk wit correct separators, decimals
'''

# ----------------------------------------------------------------------------------------------------------------------
# step 8, re-read and check dtypes again to find errors
# ----------------------------------------------------------------------------------------------------------------------
'''
now re read all data (4 dataframes - the original one without rows51, 52, 144 and the three repaired rows)
combine the three dataframes and recheck for data types (try to convert numeric attributes into float - you'll see the problems then
If you have skipped the three ruined rows just read the df without the three ruined rows and continue to check dtypes
'''


# ----------------------------------------------------------------------------------------------------------------------
# step 8, handle categorical data
# ----------------------------------------------------------------------------------------------------------------------
'''
now you can look at unique values of categorical attributes using e.g. value_counts()
this way you'll find problematic values that need recoding (e.g. AUT to AUTUMN)
Here you can also check if there is a column in which two columns are combined and split it
'''
print(df['season'].value_counts())
df.replace('spring', 'SPRING', inplace=True)
df.replace('aut', 'AUTUMN', inplace=True)
print(df['season'].value_counts())

# ----------------------------------------------------------------------------------------------------------------------
# step 9, check split columns
# ----------------------------------------------------------------------------------------------------------------------
'''
data type changes might be needed for split columns
'''


# ----------------------------------------------------------------------------------------------------------------------
# step 10, exclude duplicates
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# step 11, find outliers
# ----------------------------------------------------------------------------------------------------------------------
'''
try to use plots to find outliers "visually"
you can also try to use statistical measures to automatically exclude problematic values but be careful
'''



# ----------------------------------------------------------------------------------------------------------------------
# step 12, impute values
# ----------------------------------------------------------------------------------------------------------------------
'''
impute missing values and excluded values using the KNN-Imputer
'''


# ----------------------------------------------------------------------------------------------------------------------
# step 13, some more info on the ds
# ----------------------------------------------------------------------------------------------------------------------
'''
get the class distribution of the target variable
'''





