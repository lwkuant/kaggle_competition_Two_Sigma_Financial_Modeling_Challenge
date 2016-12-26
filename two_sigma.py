# -*- coding: utf-8 -*-
"""
kaggle competition Two Sigma Financial Modeling Challenge
"""

"""
Problem:

"""

### EDA

# load required packages
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# load the data 
import os 
os.chdir(r'D:\Dataset\kaggle_competition_Two Sigma Financial Modeling Challenge\train.h5')

df = pd.HDFStore("train.h5", "r")
df = df.get('train')
print(df.shape) # 111 variables
print(df.info())
print(df.isnull().any()) # check for NAs
print(np.sum(df.isnull().any()) ) # 106 with NAs
print(df.head())
