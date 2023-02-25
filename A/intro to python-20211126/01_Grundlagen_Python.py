# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Some Python Basics (iPython Notebook)
# =================================
# %% [markdown]
# Basics
# ------
# 
# Comments in Python:

# %%
# you can mix text and code in one place and
# run code from a Web browser

# %% [markdown]
# All you need to know about Python is here:
# 
# You don't need to specify type of a variable

# %%
a = 11


# %%
print(a)
print(type(a) is bool)
type(a)

# %% [markdown]
# You can assign several variables at once:

# %%
a, b = 1, 2
a, b


# %%
b, a = a, b
a, b

# %% [markdown]
# There is no "begin-end"! You use indentation to specify blocks. Here is simple IF statement:

# %%
if a > b:
    print("A is greater than B")
    print("x")
else:
    print("B is greater than A")

# %% [markdown]
# Types
# -----

# %%
# Integer
a = 1
print(a)

# Float
b = 1.0
print(b)

# String
c = "Hello world"
print(c)

# Unicode
d = u"Привет, мир!"
print(d)

# List (array)
e = [1, 2, 3]
print(e[2]) # 3

# Tuple (constant array)
f = (1, 2, 3)
print(f[0]) # 1

# Set
g = {1, 1, 1, 2}
print(g)

# Dictionary (hash table, hash map)
g = {1: 'One', 2: 'Two', 3: 'Three'}
print(g[1]) # 'One'

# %% [markdown]
# Loops
# -----
# %% [markdown]
# ### for

# %%
for i in range(10):
    print(i)

# %% [markdown]
# ### while

# %%
i = 0
while i < 10:
    print(i)
    i += 1

# %% [markdown]
# ### List and Enumerate

# %%
items = ['apple', 'banana', 'stawberry', 'watermelon']
# append an element
items.append('blackberry')
# removes last element
items.pop()
# insert element on the 2nd position
items.insert(1, 'blackberry')
# lenght of the list
print('Length of the list:', len(items),'\nElements:')
for item in items:
    print('- ', item)


# %%
for i, item in enumerate(items):
    print(i, item)

# %% [markdown]
# Python code style
# =================
# 
# There is PEP 8 (Python Enhancement Proposal), which contains all wise ideas about Python code style. Let's look at some of them:
# %% [markdown]
# Naming
# ------

# %%
# Variable name
my_variable = 1

# Class method and function names
def my_function():
    pass

# Constants
MY_CONSTANT = 1

# Class name
class MyClass(object):
    # 'private' variable - use underscore before a name
    _my_variable = 1

    # 'protected' variable - use two underscores before a name
    __my_variable = 1

    # magic methods
    def __init__(self):
        self._another_my_variable = 1

# %% [markdown]
# String Quotes
# -------------
# 
# PEP 8 quote:
# > In Python, single-quoted strings and double-quoted strings are the same. PEP 8 does not make a recommendation for this. Pick a rule and stick to it. When a string contains single or double quote characters, however, use the other one to avoid backslashes in the string. It improves readability.
# 
# > For triple-quoted strings, always use double quote characters to be consistent with the docstring convention in PEP 257.
# 
# My rule for single-quoted and double-quoted strings is:
# 1. Use single-quoted for keywords;
# 2. Use double-quoted for user text;
# 3. Use tripple-double-quoted for all multiline strings and docstrings.

# %%
'string'

"another string"

"""Multiline
string"""

'''
Another
multiline
string
'''

# %% [markdown]
# Some tricks
# -------------
# %% [markdown]
# Sum all elements in an array is straightforward:

# %%
sum([1,2,3,4,5])

# %% [markdown]
# However, there is no built-in function for multiplication:

# %%
#mult([1,2,3,4,5])

# %% [markdown]
# So we have to write our solution. Let's start with straightforward one:

# %%
def mult(array):
    result = 1
    for item in array:
        result *= item
    return result


# %%
mult([1,2,3,4,5])


# %%
import numpy as np
import pandas as pd
import time

# load data from a csv file using pandas (pd)
start_time = time.time()
df = pd.read_csv('01_sample_data_movies.csv', encoding='utf8')
df.head()


# %%
# some infos on the data-frame (columns, memory usage)
df.info()


# %%
# description of data-frame, including the 7-number summary
df.describe()

# %% [markdown]
# Very important references
# =====================
# 
# * PEP 8 - Style Guide for Python Code: https://www.python.org/dev/peps/pep-0008/
# * Python 3 Documentation: https://docs.python.org/3/
# %% [markdown]
# ##### Exercise: Take the following list and write a program that prints out all the elements of the list that are smaller than 5.

# %%
a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

# %% [markdown]
# ##### Exercise: Take the following two lists and write a program that returns a list that contains only the elements that are common between the lists (without duplicates). Make sure your program works on two lists of different sizes. Moreover, try to find a 1-line-solution (using sets).

# %%
import random
a = random.sample(range(1, 100), 10)
b = random.sample(range(1, 100), 12)

# %% [markdown]
# ##### Exercise: Write one line of Python that takes the following list a and makes a new list that has only the even elements of this list in it.

# %%
a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# %% [markdown]
# ##### Exercise: Implement a function that takes as input three variables, and returns the largest of the three. Do this without using the Python max() function!
# %% [markdown]
# ##### Exercise: Write a Python program to concatenate following dictionaries to create a new one.

# %%
dic1={1:10, 2:20}
dic2={3:30, 4:40}
dic3={5:50, 6:60}

# %% [markdown]
# ##### Exercise: With a given integral number n, write a program to generate a dictionary that contains (i, i*i) such that i is an integral number between 1 and n (both included). Remark: User input can be captured using the command input().
# %% [markdown]
# ##### Exercise: Generate a random number between 1 and 25 (including 1 and 25). Ask the user to guess the number, then tell them whether they guessed too low, too high, or exactly right. Remark: Import and use the random library.

# %%
# generate a random number

# generate a random number
import random
number = random.randint(1,25)

success = False
x = 0
while not success:
    print("Guess:")
    x = int(input())
    if x > number:
        print("Too high!")
    elif x < number:
        print("Too low!")
    else:
        print("Exactly right")
        success = True


# %%



# %%



