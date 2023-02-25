# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Data Manipulation
# 
# %% [markdown]
# We first import numpy and pandas libraries:
# * Pandas
# * NumPy

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# Next, we import the iris data set as a pandas data frame, and show the first few rows (the "head"):

# %%
iris = pd.read_csv('03_iris.csv')
iris.head()


# %%


# %% [markdown]
# # Selecting Columns and Rows
# %% [markdown]
# ## Select columns
# %% [markdown]
# Columns can be selected by indexing the data frame with a *list* of column names:

# %%
iris[ ['Sepal.Length', 'Sepal.Width'] ].head()

# %% [markdown]
# Columns can be droped using the `drop()` method, applied to axis 1 (= columns):

# %%
iris.drop('Species', axis = 1).head()

# %% [markdown]
# More complex filtering can be done by extracting the columns string names from the data frame, and then applying some filtering method like `startswith()` or `findall()`. The resulting indexing object (a pandas `Series`) can then be used with the `loc()` method to extract the corresponding columns.

# %%
iris.loc[:,iris.columns.str.startswith('Sepal')].head()


# %%
iris.loc[:,iris.columns.str.contains('^S.*\\.')].head()

# %% [markdown]
# ## Filter rows
# %% [markdown]
# Rows can simply be selected by specifying one or more predicates within the indexing operator:

# %%
iris[iris.Species == 'versicolor'].head()


# %%
iris[(iris['Sepal.Length'] > 5) & (iris['Sepal.Width'] > 4)]

# %% [markdown]
# # Transforming Variables
# %% [markdown]
# ## Change content
# %% [markdown]
# Modify existing variable:

# %%
iris['Sepal.Length'] = iris['Sepal.Length'].round()
iris.head()

# %% [markdown]
# Adding a new variable, based on an existing one, can be done by using the `where()` funtion from the `numpy` library. It works like vectorized ternary operator:

# %%
iris['Sepal'] = np.where(iris['Sepal.Length'] > 5, 'Long', 'Short')
iris.head()

# %% [markdown]
# ## Renaming variables
# %% [markdown]
# Renaming variables can be done using the `rename()` method. The simplest way is by specifying a dictionary of old name/new name pairs as the `columns` argument:

# %%
iris.rename(columns = {'Sepal.Length': 'Sepal_Length'}).head()

# %% [markdown]
# # Sorting and Summarizing
# %% [markdown]
# ## Sorting
# %% [markdown]
# The `sort_values()` method allows sorting according to several columns---descendoing or ascending:

# %%
iris.sort_values(by = ['Species', 'Sepal.Length'], ascending = [False, True])

# %% [markdown]
# ## Summarizing
# %% [markdown]
# ### Standard statistics
# %% [markdown]
# The simplest way to compute summary statistics is to use the `describe()` method. It computes, for each variable, count, mean, standard deviation, and all quartiles (including min/max).

# %%
iris.describe()

# %% [markdown]
# ### Individual statistics
# %% [markdown]
# These statistics can also computed individually by first selecting a variable, and then calling the corresponding method:

# %%
iris['Sepal.Length'].mean()


# %%
iris.Species.value_counts()

# %% [markdown]
# Aggregation functions can also be applied to the complete data frame:

# %%
iris.mean()

# %% [markdown]
# ## Grouping
# %% [markdown]
# An important feature is grouping. After grouping, all summary methods are applied to each group separately. In the following, we first group by the `Species` column, and then apply some summary methods:

# %%
iris.groupby('Species').mean()


# %%
iris.groupby('Species')['Sepal.Length'].describe()


# %%



