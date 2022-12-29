"""Authors: Ziang Xu, Alice Li, Shawn Lang 
  This script contains the k-fold method to return the evalutaions on different models.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statistics import mean
import estimator
import my_utils
import statistics

# This method use create, fit, and predict function in estimator to find the MAE, RMSE, and R-square for each model.
def k_fold(dataframe):
    X = dataframe.iloc[:,5:14]
    y = dataframe.iloc[:,3]
    
    kf = KFold(n_splits=5,random_state = 10,shuffle = True)
    kf.get_n_splits(X)
    
    MAE_overall = [None]*4
    MAE_naive, MAE_linear_reg, MAE_rf, MAE_FFNN = [],[],[],[]
    
    MAE_overall_train = [None]*3
    MAE_linear_reg_train, MAE_rf_train, MAE_FFNN_train = [],[],[]
    
    MAE_overall_sd = [None]*4
    MAE_naive_sd, MAE_linear_reg_sd, MAE_rf_sd, MAE_FFNN_sd = [],[],[],[]
    
    RMSE_overall = [None]*4
    RMSE_naive, RMSE_linear_reg, RMSE_rf, RMSE_FFNN = [],[],[],[]
    
    RMSE_overall_train = [None]*3
    RMSE_linear_reg_train, RMSE_rf_train, RMSE_FFNN_train = [],[],[]
    
    RMSE_overall_sd = [None]*4
    RMSE_naive_sd, RMSE_linear_reg_sd, RMSE_rf_sd, RMSE_FFNN_sd = [],[],[],[]
    
    R2_overall = [None]*4
    R2_naive, R2_linear_reg, R2_rf, R2_FFNN = [],[],[],[]
    
    R2_overall_train = [None]*3
    R2_linear_reg_train, R2_rf_train, R2_FFNN_train = [],[],[]
    
    R2_overall_sd = [None]*4
    R2_naive_sd, R2_linear_reg_sd, R2_rf_sd, R2_FFNN_sd = [],[],[],[]
    
    preprocessor = my_utils.Preprocessor()
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        preprocessor.fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        
        y_train, y_test = y[train_index], y[test_index]
        random_regr,NN_regr = estimator.create()
        random_regr,linear_regr,NN_regr = estimator.fit(random_regr,NN_regr,X_train,y_train)
        random_pred,linear_pred,NN_pred = estimator.predict(random_regr,linear_regr,NN_regr,X_test)
        random_pred_train,linear_pred_train,NN_pred_train = estimator.predict_train(random_regr,linear_regr,NN_regr,X_train)
        naive = [mean(y_test)]*len(y_test)
        
        # compute MAE of the models
        MAE_naive.append(mean_absolute_error(y_test,naive))
        MAE_linear_reg.append(mean_absolute_error(y_test, linear_pred))
        MAE_linear_reg_train.append(mean_absolute_error(y_train, linear_pred_train))
        MAE_rf.append(mean_absolute_error(y_test, random_pred))
        MAE_rf_train.append(mean_absolute_error(y_train, random_pred_train))
        MAE_FFNN.append(mean_absolute_error(y_test, NN_pred))
        MAE_FFNN_train.append(mean_absolute_error(y_train, NN_pred_train))
        
        # compute RMSE of the models
        RMSE_naive.append(mean_squared_error(y_test, naive, squared=False))
        RMSE_linear_reg.append(mean_squared_error(y_test, linear_pred, squared=False))
        RMSE_linear_reg_train.append(mean_squared_error(y_train, linear_pred_train, squared=False))
        RMSE_rf.append(mean_squared_error(y_test, random_pred, squared=False))
        RMSE_rf_train.append(mean_squared_error(y_train, random_pred_train, squared=False))
        RMSE_FFNN.append(mean_squared_error(y_test, NN_pred, squared=False))
        RMSE_FFNN_train.append(mean_squared_error(y_train, NN_pred_train, squared=False))
        
        # compute R2 of the models
        R2_naive.append(r2_score(y_test, naive))
        R2_linear_reg.append(r2_score(y_test, linear_pred))
        R2_linear_reg_train.append(r2_score(y_train, linear_pred_train))
        R2_rf.append(r2_score(y_test, random_pred))
        R2_rf_train.append(r2_score(y_train, random_pred_train))
        R2_FFNN.append(r2_score(y_test, NN_pred))
        R2_FFNN_train.append(r2_score(y_train, NN_pred_train))

    MAE_overall[0] = mean(MAE_naive)
    MAE_overall[1] = mean(MAE_linear_reg)
    MAE_overall[2] = mean(MAE_rf)
    MAE_overall[3] = mean(MAE_FFNN)
    
    MAE_overall_train[0] = mean(MAE_linear_reg_train)
    MAE_overall_train[1] = mean(MAE_rf_train)
    MAE_overall_train[2] = mean(MAE_FFNN_train)
    
    MAE_overall_sd[0] = statistics.stdev(MAE_naive)
    MAE_overall_sd[1] = statistics.stdev(MAE_linear_reg)
    MAE_overall_sd[2] = statistics.stdev(MAE_rf)
    MAE_overall_sd[3] = statistics.stdev(MAE_FFNN)
  
    RMSE_overall[0] = mean(RMSE_naive)
    RMSE_overall[1] = mean(RMSE_linear_reg)
    RMSE_overall[2] = mean(RMSE_rf)
    RMSE_overall[3] = mean(RMSE_FFNN)
    
    RMSE_overall_train[0] = mean(RMSE_linear_reg_train)
    RMSE_overall_train[1] = mean(RMSE_rf_train)
    RMSE_overall_train[2] = mean(RMSE_FFNN_train)
    
    RMSE_overall_sd[0] = statistics.stdev(RMSE_naive)
    RMSE_overall_sd[1] = statistics.stdev(RMSE_linear_reg)
    RMSE_overall_sd[2] = statistics.stdev(RMSE_rf)
    RMSE_overall_sd[3] = statistics.stdev(RMSE_FFNN)
    
    R2_overall[0] = mean(R2_naive)
    R2_overall[1] = mean(R2_linear_reg)
    R2_overall[2] = mean(R2_rf)
    R2_overall[3] = mean(R2_FFNN) 
    
    R2_overall_train[0] = mean(R2_linear_reg_train)
    R2_overall_train[1] = mean(R2_rf_train)
    R2_overall_train[2] = mean(R2_FFNN_train) 
    
    R2_overall_sd[0] = statistics.stdev(R2_naive)
    R2_overall_sd[1] = statistics.stdev(R2_linear_reg)
    R2_overall_sd[2] = statistics.stdev(R2_rf)
    R2_overall_sd[3] = statistics.stdev(R2_FFNN) 

                       
    return MAE_overall,RMSE_overall,R2_overall,MAE_overall_train,RMSE_overall_train,R2_overall_train,MAE_overall_sd,RMSE_overall_sd,R2_overall_sd