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
print(np.sum(df.isnull().any()) ) # 106 with NAs
print(df.head())

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

