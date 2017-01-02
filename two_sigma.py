# -*- coding: utf-8 -*-
"""
kaggle competition Two Sigma Financial Modeling Challenge
"""

"""
Problem:
(1) build a model for each category of assets?
(2) use unsupervised methods to do some grouping 
(3) how to deal with NAs for each column? (ex. mean for each product id)
(4) filter out the rows with y too large or small
"""

"""
Preprocessing Steps:
(1) use the highly correlated columns (not yet)
(2) fill the NAs with mean of each id and then the mean of all values
(3) filter out the ys that are too large or small
"""

### EDA

## load required packages
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

## load the data 
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

## column type 
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

## check the distribution of the y
print(df.y.isnull().any()) # no NA

import matplotlib
matplotlib.style.use('ggplot')   
fig, axes = plt.subplots(figsize=[15, 15])
axes.tick_params(labelsize=15)
sns.distplot(df.y, bins=100, kde=False)
axes.set_title('Frequency Distribution of y', fontsize=30)
axes.set_xlabel('Value', fontsize=25)
axes.set_ylabel('Count', fontsize=25)
plt.savefig('y_distribution.png', dpi=300)

print(df.y.mean()) # mean is close to 0, 0.00022s
print(df.y.std(ddof=0)) # std is 0.022
from scipy.stats import skew
print(skew(df.y.values)) # skewness is 0.22
# the distribution of y is nearly normal distributed
# the relative high frequencies on both of the ends are worth studying

## check the distribution of the timestamp
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

## The number of NAs in each variable 
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

## Correlation between features and the target
from scipy.stats import pearsonr
col_name = df.columns
col_name_feat = col_name[1:-1]

cor = np.array([pearsonr(df.ix[:, [col]].dropna().values,
                         df.ix[:, ['y']].ix[df.ix[:, [col]].dropna().index, :].values)[0] for col in col_name_feat])

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

## filter out the features with high correlatino and significance
sig = np.array([pearsonr(df.ix[:, [col]].dropna().values,
                         df.ix[:, ['y']].ix[df.ix[:, [col]].dropna().index, :].values)[1] for col in col_name_feat])

sig_good = sig[sig<=0.5]
cor_good = cor[sig<=0.5]

sort_ind = np.argsort(np.abs(cor_good))[::-1]

col_name_feat_sort = col_name_feat[(sig<=0.5).ravel()][sort_ind]

good_cor = list(col_name_feat_sort)[:5]
print(good_cor)

# visualize the correlation between features
temp_df = df[good_cor]
corrmat = temp_df.corr(method='pearson')
fig, axes = plt.subplots(figsize=(8, 8))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
# from the graph above, we find out that technical_27 is highly correlated with
# technical_19
# the technical_27 is more strongly correlated than technical_27
# from the graph above, we can also find out that technical_30 is well correlated wiyh 
# technical_20
# the technical_20 is more strongly correlated than technical_30

## how many cateegories of ids are there?
print(len(df.id.value_counts())) # 1424 categories

## filter out a new data frame which only has needed columns (including id, 
## highly correlated variables and y)
df_filtered = df.ix[:, ['id', 'technical_20', 'fundamental_11', 'technical_30',
                        'technical_19', 'y']]
print(df_filtered.info())

## fill each column with rows with NAs using the mean for each id 
for col in ['technical_20', 'fundamental_11', 'technical_30', 'technical_19']:
    #median_dict = dict(df_filtered.groupby(['id'])[col].median())    
    #na_ind = df_filtered[col].isnull()
    #df_filtered[col][na_ind] = df_filtered['id'][na_ind]
    #df_filtered[col][na_ind] = df_filtered[col][na_ind].map(median_dict)
    df_filtered[col].fillna(df_filtered[col].median(), inplace=True)
print(df_filtered.isnull().any())    
        
## find out where are the borders of extreme large and small values
cum_count = np.histogram(df_filtered.y, bins=100, range=[-0.1, 0.1])[0].cumsum()
bin_name = np.histogram(df_filtered.y, bins=100, range=[-0.1, 0.1])[1]
bin_ind = list(range(100))
fig, axes = plt.subplots(figsize=[15, 15])
axes.plot(bin_ind, cum_count)
axes.set_ylabel('Frequency')

print(np.histogram(df_filtered.y, bins=100, range=[-0.1, 0.1])[0]) # index: small:6, 
# large:96
min_border = np.histogram(df_filtered.y, bins=100, range=[-0.1, 0.1])[1][7]
max_border = np.histogram(df_filtered.y, bins=100, range=[-0.1, 0.1])[1][96]

min_ind = df_filtered.y<min_border
max_ind = df_filtered.y>=max_border
border_ind = min_ind|max_ind

## assign the labels to rows with y falling outside of extreme borders
df_filtered['out'] = border_ind
print(df_filtered['out'].value_counts())
df_filtered['out'] = df_filtered['out'].astype(int)
print(df_filtered['out'].value_counts())

## compare the distribution of independent variables between two types of y values
fig, axes = plt.subplots(figsize=[15, 15])
sns.boxplot(x='out', y='technical_20', data=df_filtered)

fig, axes = plt.subplots(figsize=[15, 15])
sns.boxplot(x='out', y='fundamental_11', data=df_filtered)

fig, axes = plt.subplots(figsize=[15, 15])
sns.boxplot(x='out', y='technical_27', data=df_filtered) # looks significant
# statistical test: anova
from scipy.stats import f_oneway
print(f_oneway(df_filtered['technical_27'][border_ind], df_filtered['technical_27'][~border_ind]))
# looks like the the column 'technical_27' is a determinant for types of y

## compare the distribution of independent variables between two types of out of border (y) values
from copy import deepcopy
df_y_filtered = deepcopy(df_filtered)
df_y_filtered = df_y_filtered.ix[border_ind, :]
ind_large = df_y_filtered.y>=max_border
df_y_filtered['large'] = ind_large.astype(int)

fig, axes = plt.subplots(figsize=[15, 15])
sns.boxplot(x='large', y='technical_20', data=df_y_filtered)

fig, axes = plt.subplots(figsize=[15, 15])
sns.boxplot(x='large', y='fundamental_11', data=df_y_filtered)

fig, axes = plt.subplots(figsize=[15, 15])
sns.boxplot(x='large', y='technical_27', data=df_y_filtered)
# cannot not find significant relationship

## build model on the data with y within the border 
df_y_filtered = df_filtered.ix[~border_ind, :]
df_y_filtered.drop(['id', 'out'], axis=1, inplace=True)


### Modeling 

## Random Forest
from sklearn.cross_validation import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(df_y_filtered.ix[:, [0, 1, 2]].values,
                                          df_y_filtered.ix[:, ['y']].values,
                                            test_size=0.2)

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=50)
import timeit
start_time = timeit.default_timer()
print(cross_val_score(model, X_tr, y_tr, cv=5))
elapsed = timeit.default_timer() - start_time
print(elapsed) # 4013.717750154111

model.fit(X_tr, y_tr)
print('done')
print(model.score(X_tr, y_tr))
print(model.score(X_te, y_te))

## Ridge Regression
from sklearn.cross_validation import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(df_y_filtered.ix[:, [0, 1, 2]].values,
                                          df_y_filtered.ix[:, ['y']].values,
                                            test_size=0.2)

from sklearn.linear_model import Ridge
import timeit
start_time = timeit.default_timer()
model = Ridge(alpha=1)
print(cross_val_score(model, X_tr, y_tr, cv=10))
elapsed = timeit.default_timer() - start_time
print(elapsed) 

model.fit(X_tr, y_tr)

print(model.score(X_te, y_te))

## Linear Regression
from sklearn.cross_validation import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(df_y_filtered.ix[:, [0, 1, 2]].values,
                                          df_y_filtered.ix[:, ['y']].values,
                                            test_size=0.2)

from sklearn.linear_model import LinearRegression
import timeit
start_time = timeit.default_timer()
model = LinearRegression()
print(cross_val_score(model, X_tr, y_tr, cv=10))
elapsed = timeit.default_timer() - start_time
print(elapsed) 

model.fit(X_tr, y_tr)

print(model.score(X_te, y_te))


### Linear Regression Test
import os 
os.chdir(r'D:\Dataset\kaggle_competition_Two Sigma Financial Modeling Challenge\train.h5')

with pd.HDFStore("train.h5", "r") as train:
    df = train.get("train")
    
import os 
os.chdir(r'D:\Project\kaggle_competition_Two_Sigma_Financial_Modeling_Challenge')

df_filtered = df.ix[:, ['id', 'technical_20', 'fundamental_11', 'technical_30',
                        'y']]

df_filtered.fillna(df_filtered.median(), inplace=True)                

min_border = np.histogram(df_filtered.y, bins=100, range=[-0.1, 0.1])[1][7]
max_border = np.histogram(df_filtered.y, bins=100, range=[-0.1, 0.1])[1][96]

min_ind = df_filtered.y<min_border
max_ind = df_filtered.y>=max_border
border_ind = min_ind|max_ind

df_y_filtered = df_filtered.ix[~border_ind, :]

df_y_filtered.drop(['id'], axis=1, inplace=True)     

from sklearn.cross_validation import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(df_y_filtered.ix[:, [0, 1, 2]].values,
                                          df_y_filtered.ix[:, ['y']].values,
                                            test_size=0.2)

from sklearn.linear_model import LinearRegression
import timeit
start_time = timeit.default_timer()
model = LinearRegression()
print(cross_val_score(model, X_tr, y_tr, cv=10))
elapsed = timeit.default_timer() - start_time
print(elapsed)  