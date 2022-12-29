"""Authors: Ziang Xu, Alice Li, Shawn Lang 
  This script contains the class for preprocessing and functions, including checking datasets, adjusting feature types, and performing k best selection.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.features_selected = ()
        self.scaler = preprocessing.StandardScaler()
        self.feature_means = ()
        self.feature_stdev = ()
        
    def fit(self, data):
        self.selected_features = data.columns
        self.scaler = self.scaler.fit(data[self.selected_features])
        self.feature_means = data.mean()
        self.feature_stdev = data.std()
    
    def transform(self, data):        
        data = data[self.selected_features]
        data = self.scaler.transform(data)
        return data
        
    def remove_features_near_zero_variance(self, dataset, threshold=1e-4):
        numeric_columns = dataset.select_dtypes('float64').columns
        columns_to_drop = []
        for column in numeric_columns:
            if dataset[column].std()**2 < threshold:
                columns_to_drop.append(column)
        dataset = dataset.drop(columns=columns_to_drop)
        return dataset        
        
    def remove_outliers(self, dataset, num_std=3):
        numeric_columns = dataset.select_dtypes('float64').columns
        for column in numeric_columns: 
            mean = dataset[column].mean()
            sd = dataset[column].std() 
            dataset = dataset[(dataset[column] <= mean + (num_std * sd))]
            dataset = dataset[(dataset[column] >= mean - (num_std * sd))]
        return dataset       
    
    def remove_highly_correlated_features(self, dataset, threshold=0.9):
        numeric_columns = dataset.select_dtypes('float64').columns
        correlation_matrix = dataset[numeric_columns].corr().abs()
        columns_to_drop = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    colname = correlation_matrix.columns[i]

        dataset = dataset.drop(columns=columns_to_drop)
        return dataset    
    
def check_datasets(dir_path, index_col, transpose_flag):
   file_count = 0
   for filename in os.listdir(dir_path):
      if filename.startswith('.'):
         continue
      file = os.path.join(dir_path, filename)
      if os.path.isfile(file):
         if index_col != None:
            dataset = pd.read_csv(file, index_col=index_col, encoding='utf-8')
         else:
            dataset = pd.read_csv(file)
         if transpose_flag:
            dataset = dataset.T
         file_count = file_count + 1

         if file_count == 1:
            expected_nrows = dataset.shape[0]
            expected_ncols = dataset.shape[1]
            row_names = dataset.index
            col_names = dataset.columns

         if dataset.isnull().values.any() == True:
            print('Missing values in ' + file)
         if dataset.duplicated().any():
            print('Duplicated rows in ' + file)

   print('Total number of files ' + str(file_count))
   

def adjust_feature_types(dataset):
    numeric_columns = dataset.select_dtypes('number').columns
    for column in numeric_columns:
        dataset = dataset.astype({column: 'float64'})
    return dataset

def k_best_selection(X_clf, Y_clf, k):
    np.seterr(invalid='ignore')
    selector = SelectKBest(f_regression, k=20)
    selector.fit(X_clf, Y_clf)
    cols = selector.get_support(indices=True)
    return cols

def run_linear_regression_tests(dataset):
    numeric_columns = dataset.select_dtypes('float64').columns
    for column in numeric_columns:
        target = dataset[column].to_numpy()
        features = dataset[numeric_columns].drop([column], axis=1)
        features = np.array(features)
        regression_model = LinearRegression().fit(features, target)
        print('Target: ' + column + 
              '\tFitted R2: ' + str(round(regression_model.score(features, target), 2)))