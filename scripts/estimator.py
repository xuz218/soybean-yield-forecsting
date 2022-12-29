"""Authors: Ziang Xu, Alice Li, Shawn Lang 
  This script contains create, fit, and predict funtions to do the estimation task.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.preprocessing import StandardScaler

# This function create the random forest model and the FFNN model with some basic settings.
def create():
    mdl = RandomForestRegressor(max_depth=5, random_state=0)
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(32, kernel_initializer='normal',input_dim = 9, activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='relu'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    return mdl,NN_model

# This function fit the training model and return the model that has been trained.
def fit(random_regr,NN_regr,X_train,y_train):
    random_regr = random_regr.fit(X_train, y_train)
    linear_regr = LinearRegression().fit(X_train, y_train)
    NN_regr.fit(X_train, y_train, epochs=30, batch_size=15, validation_split = 0.2)
    return random_regr,linear_regr, NN_regr

# This function return the predicted value from each model with test data input.
def predict(random_regr,linear_regr,NN_regr,X_test):
    random_pred = random_regr.predict(X_test)
    linear_pred = linear_regr.predict(X_test)
    NN_pred = NN_regr.predict(X_test)
    return random_pred,linear_pred,NN_pred

def predict_train(random_regr,linear_regr,NN_regr,X_train):
    random_pred_train = random_regr.predict(X_train)
    linear_pred_train = linear_regr.predict(X_train)
    NN_pred_train = NN_regr.predict(X_train)
    return random_pred_train,linear_pred_train,NN_pred_train

