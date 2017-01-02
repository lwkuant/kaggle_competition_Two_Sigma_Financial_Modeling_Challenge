import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))


train_data = observation.train
#print(train_data.head())


# Columns for training
#cols = [col for col in train_data_cleaned.columns if "technical_" in col]
cols = ['technical_20', 'fundamental_11', 'technical_30']
print(cols)

low_y_cut = -0.086
high_y_cut = 0.092

y_values_within = ((train_data['y'] > low_y_cut) & (train_data['y'] <high_y_cut))

train_cut = train_data.loc[y_values_within,:]

# Fill missing values
median_vals = train_cut.median()
train_cut.fillna(median_vals,inplace=True)




x_train = train_cut[cols]
y = train_cut["y"]

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

print(x_train.shape)
print(y.shape)
lr_model.fit(x_train.values,y.values)



while True:
    observation.features.fillna(median_vals, inplace=True)
    x_test = observation.features[cols].values
    ypred = lr_model.predict(x_test)
    observation.target.y = ypred

    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
    
    
# Public score: 0.01914269597332126
# LB: 0.0092780