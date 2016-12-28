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

with pd.HDFStore("train.h5", "r") as train:
    df = train.get("train")
print(df.shape) # 1710756 rows, 111 variables
print(df.info())
print(df.isnull().any()) # check for NAs
print(np.sum(df.isnull().any())) # 106 with NAs
print(df.head())
print(df.describe())

import os 
os.chdir(r'D:\Project\kaggle_competition_Two_Sigma_Financial_Modeling_Challenge')

# column type 
col_name = df.columns

from collections import defaultdict
col_type = defaultdict()
col_type['derived'] = 0
col_type['fundamental'] = 0
col_type['technical'] = 0

for type_name in col_name:
    type_name_split = type_name.split('_')[0]
    if type_name_split == 'derived':
        col_type['derived'] += 1
    elif type_name_split == 'fundamental':
        col_type['fundamental'] += 1
    elif type_name_split == 'technical':
        col_type['technical'] += 1
    else:
        continue

for type_name in list(col_type.keys()):
    print('%s has    %d columns' %(type_name, col_type[type_name]))

# check the distribution of the y
print(df.y.isnull().any()) # no NA

import matplotlib
matplotlib.style.use('ggplot')   
fig, axes = plt.subplots(figsize=[15, 15])
axes.tick_params(labelsize=15)
sns.distplot(df.y, bins=100, kde=False)
axes.set_title('Frequency Distribution of y', fontsize=30)
axes.set_xlabel('Value', fontsize=25)
axes.set_ylabel('Count', fontsize=25)

print(df.y.mean()) # mean is close to 0, 0.00022s
print(df.y.std(ddof=0)) # std is 0.022
from scipy.stats import skew
print(skew(df.y.values)) # skewness is 0.22
# the distribution of y is nearly normal distributed
# the relative high frequencies on both of the ends are worth studying

# check the distribution of the timestamp
print(df.timestamp.isnull().any()) # no NA

import matplotlib
matplotlib.style.use('ggplot')   
fig, axes = plt.subplots(figsize=[15, 15])
axes.tick_params(labelsize=15)
sns.distplot(df.timestamp, bins=100, kde=False)
axes.set_title('Frequency Distribution of timestamp', fontsize=30)
axes.set_xlabel('Value', fontsize=25)
axes.set_ylabel('Count', fontsize=25)
# there seemsto have some patterns 

# The number of NAs in each variable 
num_NAs = df.isnull().sum()
num_NAs_count = num_NAs.values
num_NAs_name = num_NAs.index

import matplotlib 

matplotlib.style.use('ggplot') 
fig, axes = plt.subplots(figsize=[15, 60])
#axes.bar(list(range(len(num_NAs_name))), num_NAs_count, orientation='vertical')
axes.tick_params(labelsize=15)
axes.barh(list(range(len(num_NAs_name))), num_NAs_count, color='#2E8B57')
axes.set_title('NA frequency for each variable', fontsize=30)
axes.set_yticks(np.arange(len(num_NAs_name))+0.4)
axes.set_yticklabels(num_NAs_name)
axes.set_xlabel('Frequency', fontsize=25)
axes.set_ylabel('Variable', fontsize=25)
#plt.savefig('NA_frequency.png', dpi=500)
# there are some very large numbeers of NAs
# we have to condier NAs processing carefully

# Correlation between features and the target
from scipy.stats import pearsonr
col_name = df.columns
col_name_feat = col_name[1:-1]

cor = np.array([pearsonr(df.ix[:, [col]].dropna().values,
                         df.ix[:, ['y']].ix[df.ix[:, [col]].dropna().index, :].values)[0] for col in col_name])

import matplotlib 

matplotlib.style.use('ggplot') 
fig, axes = plt.subplots(figsize=[15, 60])
#axes.bar(list(range(len(num_NAs_name))), num_NAs_count, orientation='vertical')
axes.tick_params(labelsize=15)
axes.barh(list(range(len(col_name_feat))), cor, color='#2E8B57')
axes.set_title('Correlation with y for each variable', fontsize=30)
axes.set_yticks(np.arange(len(col_name_feat))+0.4)
axes.set_yticklabels(col_name_feat)
axes.set_xlabel('Correlation', fontsize=25)
axes.set_ylabel('Variable', fontsize=25)
#plt.savefig('Correlation_with_y.png', dpi=500)

