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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.datasets import load_wine
from scipy.stats import norm
from io import StringIO

########################################################################################################################
# Solution
########################################################################################################################


'''
PLease note:
- the structure below can help you, but you can also completely ignore it
'''

# set pandas options to make sure you see all info when printing dfs
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

'''
I put lots of statements in comments so i don't read messed up dfs all the time
'''

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 1, just try it
# ----------------------------------------------------------------------------------------------------------------------
'''
start by just loading the data
'''
# df = pd.read_csv('wine_exercise.csv', sep=';')

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 2, use skip rows
# ----------------------------------------------------------------------------------------------------------------------
'''
skip row to ignore text inside the file
'''
# df = pd.read_csv('wine_exercise.csv', sep=';', skiprows=1)

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 3, use skipfooter
# ----------------------------------------------------------------------------------------------------------------------
'''
the footer is also a problem, skip it with skipfooter
'''
# df = pd.read_csv('wine_exercise.csv', sep=';', skiprows=1, skipfooter=1, engine='python')
# df.info()
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
# df = pd.read_csv('wine_exercise.csv', sep=';', skiprows=1, skipfooter=1, engine='python')
# df.astype({'alcohol': 'float64'}).dtypes
# the types can't be converted because some cells are messed up.

# exclude messed up lines manually
# df = pd.read_csv('wine_exercise.csv', sep=';', decimal='.', skiprows=[0, 52, 53, 144, 168], skipfooter=1, engine='python', na_values='missing')

# or automatically
'''
This just checks if commas or white spaces are found. this could be improved by looking for a list of
wanted or unwanted separators.
'''
file = open('wine_exercise.csv', 'r')
counter = 0
# these row indices will be skipped
skip_lines = []
# this is the content to later be fixed
repair_lines = []
for line in file:
    if line.find(",") != -1 or line.find(" ") != -1:
        skip_lines.append(counter)
        repair_lines.append(line)
    counter += 1
file.close()
df = pd.read_csv('wine_exercise.csv', sep=';', decimal='.', skiprows=skip_lines, engine='python', na_values='missing')
df.astype({'alcohol': 'float64', 'ash': 'float64'}).dtypes
# now the conversion works
# df.info()
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 6, exclude the three problematic rows
# ----------------------------------------------------------------------------------------------------------------------
'''
the three rows are completely ruined and can only be fixed in isolation
you can read the dataset an skip these rows
'''
# as done above

# ----------------------------------------------------------------------------------------------------------------------
# step 7, handle rows separately
# ----------------------------------------------------------------------------------------------------------------------
'''
If this is too much data dirt for you continue without handling these three rows (continue with step 8)
Otherwise you can follow the workflow indicated below (steps 7.1, 7.2, 7.3, 7.4)
'''
# i'd like to do that but i might be running out of time

# ----------------------------------------------------------------------------------------------------------------------
# step 7.1, first get the column names from the df
'''
get column names so you can assign them to the single rows you did read
'''
columns = df.columns.values.tolist()

# ----------------------------------------------------------------------------------------------------------------------
# step 7.2, handle row 52
'''
read only row 52 and repair it in isolation
write it to disk wit correct separators, decimals
'''
# correct sep
repair_lines[1] = repair_lines[1].replace(",", ";")
csvStringIO = StringIO(repair_lines[1])
repaired = pd.read_csv(csvStringIO, sep=";", header=None, names=columns)
df = pd.concat([df, repaired], ignore_index=True)
# ----------------------------------------------------------------------------------------------------------------------
# step 7.3, handle row 53
'''
read only row 53 and repair it in isolation
write it to disk wit correct separators, decimals
'''
# correct dec
repair_lines[2] = repair_lines[2].replace(",", ".")
csvStringIO = StringIO(repair_lines[2])
repaired = pd.read_csv(csvStringIO, sep=";", header=None, names=columns)
df = pd.concat([df, repaired], ignore_index=True)

# ----------------------------------------------------------------------------------------------------------------------
# step 7.4, handle row 144
'''
read only row 144 and repair it in isolation
write it to disk wit correct separators, decimals
'''
# correct sep
repair_lines[3] = repair_lines[3].replace(",", ";")
csvStringIO = StringIO(repair_lines[3])
repaired = pd.read_csv(csvStringIO, sep=";", header=None, names=columns)
df = pd.concat([df, repaired], ignore_index=True)

# correct sep
repair_lines[4] = repair_lines[4].replace(" ", ";")
repair_lines[4] = repair_lines[4].replace(";;", ";")
csvStringIO = StringIO(repair_lines[4])
repaired = pd.read_csv(csvStringIO, sep=";", header=None, names=columns)
df = pd.concat([df, repaired], ignore_index=True)
# ----------------------------------------------------------------------------------------------------------------------
# step 8, re-read and check dtypes again to find errors
# ----------------------------------------------------------------------------------------------------------------------
'''
now re read all data (4 dataframes - the original one without rows51, 52, 144 and the three repaired rows)
combine the three dataframes and recheck for data types (try to convert numeric attributes into float - you'll see the problems then
If you have skipped the three ruined rows just read the df without the three ruined rows and continue to check dtypes
'''
# df.info()
# ----------------------------------------------------------------------------------------------------------------------
# step 8, handle categorical data
# ----------------------------------------------------------------------------------------------------------------------
'''
now you can look at unique values of categorical attributes using e.g. value_counts()
this way you'll find problematic values that need recoding (e.g. AUT to AUTUMN)
Here you can also check if there is a column in which two columns are combined and split it
'''

# fix wrong season labels
# print(df['season'].value_counts())
df.replace('spring', 'SPRING', inplace=True)
df.replace('aut', 'AUTUMN', inplace=True)
# print(df['season'].value_counts())

# print(df['color'].value_counts())
# if there are only two colors, do I need to split the column to is_1 or is_0 or do I leave it that way?
# I think it's ok to leave it that way, binary discrete values seem pretty straight forward

# ----------------------------------------------------------------------------------------------------------------------
# step 9, check split columns
# ----------------------------------------------------------------------------------------------------------------------
'''
data type changes might be needed for split columns
'''
# split at '-' and put into new columns
df[['country', 'years-old']] = df.loc[:, 'country-age'].str.split("-", expand=True)
# remove 'years' suffix from every years-old cell
df['years-old'] = df.loc[:, 'years-old'].str.replace("years", "")
# drop old column
df.drop('country-age', axis=1, inplace=True)
# convert years-old column to int
df['years-old'] = df['years-old'].astype('int')

# df.info()
# ----------------------------------------------------------------------------------------------------------------------
# step 10, exclude duplicates
# ----------------------------------------------------------------------------------------------------------------------
df.drop_duplicates(inplace=True)

# ----------------------------------------------------------------------------------------------------------------------
# step 11, find outliers
# ----------------------------------------------------------------------------------------------------------------------
'''
try to use plots to find outliers "visually"
you can also try to use statistical measures to automatically exclude problematic values but be careful
'''

# get column names of numerical columns
columns = df.select_dtypes(include=np.number).columns.values.tolist()
# no need to plot the target
columns.remove("target")

# scale values, so they can be plotted in one figure meaningfully
scaler = MinMaxScaler()
df[columns] = scaler.fit_transform(df[columns])

# plot
boxplot = df.boxplot(column=columns)
plt.xticks(rotation=-90)
plt.subplots_adjust(bottom=0.4)
plt.show()

'''
# plotted each column separately at first but that's kind of chaotic in the SciView
for label in columns:
    boxplot = df.boxplot(column=label)
    plt.show()
'''

'''
these look suspicious:
    malic_acid
    total_phenols
    
these have some outliers but don't look too weird, i'd say:
    ash
    alcalinity_of_ash
    magnesium
    proanthocyanins
    color_intensity
    hue
'''

# look at the suspicious ones again
boxplot = df.boxplot(column=['malic_acid'])
plt.show()
boxplot = df.boxplot(column=['total_phenols'])
plt.show()

# malic_acid: it seems like missing values are encoded as 0 (or -999 before scaling)
df['malic_acid'] = df.loc[:, 'malic_acid'].replace(0, np.nan)
boxplot = df.boxplot(column=['malic_acid'])
plt.show()
# much better

# the proline boxplot doesn't look to weird, but when looking at the dataframe the zero values seem off
# it also says in the instructions that the proline nan values are to be considered
df['proline'] = df.loc[:, 'proline'].replace(0, np.nan)

# total_phenols: two values (indices 163, 164) seem to be missing a comma (values above 100 instead of roughly 1)
# (typing error?)
# inverse scaling to find wrong phenols
df[columns] = scaler.inverse_transform(df[columns])
# find the wrong phenols
fix_phenols = df.query('total_phenols > 100')
# replace in old df with scaled values
df.loc[df.total_phenols.isin(fix_phenols.total_phenols), ['total_phenols']] = fix_phenols[['total_phenols']] / 100
# or replace with nan to later impute them
# df.loc[df.total_phenols.isin(fix_phenols.total_phenols), ['total_phenols']] = np.nan
# plot again
boxplot = df.boxplot(column=['total_phenols'])
plt.show()
# much better

# magnesium ?
# am confused, statistics.

# ----------------------------------------------------------------------------------------------------------------------
# step 12, impute values
# ----------------------------------------------------------------------------------------------------------------------
'''
impute missing values and excluded values using the KNN-Imputer
'''
# you should scale before imputing

original = load_wine()
# print(original['feature_names'])
imputer = KNNImputer(n_neighbors=5)
df[original['feature_names']] = imputer.fit_transform(df[original['feature_names']])

# ----------------------------------------------------------------------------------------------------------------------
# step 13, some more info on the ds
# ----------------------------------------------------------------------------------------------------------------------
'''
get the class distribution of the target variable
'''
class_balancing = df[["target"]].groupby(["target"]).size()

# magnesium group by color
magnesium_color = df.groupby('color')['magnesium'].agg(['mean', 'median', 'min', 'max'])
# group by color and calculates min, max, median and mean on the magnesium column

# (3) magnesium, i'm confused?
