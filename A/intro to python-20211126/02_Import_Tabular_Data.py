# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Reading tabular data
# %% [markdown]
# We start importing the pandas library, and set the number of maximally displayed rows to 5:

# %%
import pandas as pd
pd.options.display.max_rows = 5

# %% [markdown]
# ## Data with delimiters
# %% [markdown]
# First, we will import some data on flowers from the provided data sets. Let's have a look at the file:

# %%
# for Linux: !head 02_text1.csv
get_ipython().system('type 02_text1.csv')

# %% [markdown]
# The first row contains the header (column names), and the columns are separated by a colon. The first 4 columns are floating point numbers, the last column is a string.
# %% [markdown]
# The pandas read method `read_csv()` tries to guess all these settings:

# %%
pd.read_csv('02_text1.csv')

# %% [markdown]
# If some guess goes wrong, the parameters can be specified explicitely:

# %%
pd.read_csv("02_text1.csv", sep = ",", header = 0,
            dtype = {'Sepal.Length' : float, 'Species' : str})

# %% [markdown]
# ## Fixed-width data
# %% [markdown]
# Text columns can be a mess when not properly quoted:

# %%
# for Linux: !cat 02_text2.txt
get_ipython().system('type 02_text2.txt')

# %% [markdown]
# In this case, we need to specify the column widths, or start/end of the columns.

# %%
data = pd.read_fwf('02_text2.txt', widths = [13, 35, 15], names = ['Name', 'Address', 'Telephone'], encoding = "UTF-8")
data

# %% [markdown]
# # Separating / Joining columns
# %% [markdown]
# In the example above, we need some further cleaning: the address should be separated into Street, ZIP code and City. We will try to split the street from the rest using the colon, and then separate ZIP and city with the space.
# %% [markdown]
# First, we replace the semi-colon with a colon:

# %%
data.Address = data.Address.str.replace(';', ',')
data

# %% [markdown]
# Now, we extract the address strings as a list, and use the `partition()` method to separate the data:

# %%
parts = data.Address.str.partition(', ')
parts

# %% [markdown]
# This results in a data frame with the three columns. We add the firt column (address) as a new `Street` column:

# %%
data['Street'] = parts[0]
data

# %% [markdown]
# Then, we further split the third column...

# %%
tmp = parts[2].str.partition(' ')
tmp

# %% [markdown]
# ... and add the result as `ZIP` and `City` columns. Finally, we delete the old `Address` column:

# %%
data['ZIP'] = tmp[0]
data['City'] = tmp[2]
data.drop('Address', axis = 1, inplace = True)
data

# %% [markdown]
# Recombining columns is easy, using the `+` operator:

# %%
data['Address'] = data.ZIP + " " + data.City + ", " + data.Street
data.drop(['ZIP', 'City', 'Street'], axis = 1)

# %% [markdown]
# # Wide and Long format
# %% [markdown]
# Consider the following `USArrests` data about the arrests for certain crimes in various US states:

# %%
USArrests = pd.read_csv("02_USArrests.csv")
USArrests

# %% [markdown]
# The colums `Murder`, `Assault` and `Rape` are three different types of crime. A cleaner structure would be a string column `Crime`, along with a numeric column `Arrests`:

# %%
US_long = USArrests.melt(id_vars = ['State'], var_name = 'Crime', value_name='Arrests')
US_long

# %% [markdown]
# This representation is called *long* format. To transform it back to the initial *wide* format, we use the `pivot()` method:

# %%
US_long.pivot(index = 'State', columns = 'Crime', values = 'Arrests')

# %% [markdown]
# # Missing data
# %% [markdown]
# Consider the following data on passengers of the Titanic disaster:

# %%
# for Linux: !cat 02_text3.txt
get_ipython().system('type 02_text3.txt')

# %% [markdown]
# The Class and Sex labels have not been repeated in each row, and there are two missing values (`??`) for Children in the Crew Class. When we read in the data using `read_fwf()`, we get:

# %%
data = pd.read_fwf('02_text3.txt', na_values='??')
data

# %% [markdown]
# 
# Both empty cells and missing values are represented by the `NaN` symbol. To clean the data, we need to replace `NaN` in the `Class` and `Sex` columns with the last known label from the top, and to handle the "real" missing values suitably. We start handling the missings in the `Class` and `Sex` columns using the `fillna()` method:

# %%
data[['Class', 'Sex']] = data[['Class', 'Sex']].fillna(method = "ffill")
data

# %% [markdown]
# The remaining `NaN` values could either be removed, or replaced by some sensible value. If the values were missing "completely at random", we could, e.g., use the mean of the other values. In this case, however, these values are not missing at random: Crew members clearly were all adult, so the correct replacement value is 0.
# %% [markdown]
# To filter all rows with missing values, we can use the `isna()` method, indicating, for each cell, whether the value is missing or not. Applying `any()` to the *transposed* table will give us, for each row, whether at least one value is missing:

# %%
data.isna().T.any()

# %% [markdown]
# Thus, all incomplete rows are given by:

# %%
data[data.isna().T.any()]

# %% [markdown]
# These could be removed using `dropna()`:

# %%
data.dropna()

# %% [markdown]
# ... or replaced with 0 again using `fillna()`:

# %%
data.fillna(0, inplace = True)
data

# %% [markdown]
# __Final note:__ because `NaN` is only available for floating point values, the data type of `Survived` is incorrect. To fix this, we can use:

# %%
data.Survived = data.Survived.astype(int)
data


# %%



