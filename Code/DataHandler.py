#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import numpy as np

import pandas as pd

import json

pd.options.mode.chained_assignment = None  # default='warn'


##########################
## Data Import & Handling
##########################

class DataHandler(object):
    
    def __init__(self):
        self.data = None
        self.data_loss_band = None
        self.feature_columns = None
        self.target_columns = None
        self.X = None
        self.y = None
        self.X_scaler = None
        self.y_scaler = None
        self.features = None
                
    def loadData(self, file, colRemove = None, skiprows = 0):
        self.data = pd.read_csv(file, skiprows = skiprows)
        self.preProcessData(colRemove = colRemove)

    def preProcessData(self, colRemove = None):
        
        if 'Package Loss (%)' in self.data.columns and 'Bandwidth (Kbps)' in self.data.columns:
            self.data_loss_band = self.data[['Package Loss (%)', 'Bandwidth (Kbps)']]
            self.data_loss_band = self.data_loss_band.dropna()
        
        self.data = DataHandler.cleanUpData(self.data, colRemove)

        # Extract values
        self.feature_columns = [c for c in self.data.columns.values if c not in ['MOS']]
        self.target_columns = [c for c in self.data.columns.values if c in ['MOS']]
        
        # If there are no labels default to last column
        if len(self.target_columns) == 0:
            self.target_columns = self.data.columns.values[-1]
        
        self.features = len(self.feature_columns)
        
        # Extract relevant columns from data frame as numpy arrays
        self.X = self.data[self.feature_columns].values
        self.y = self.data[self.target_columns].values
        
        # Reshape target data into a 2d array [[N x 1]]
        self.y = self.y.reshape(-1, 1)
       
    @staticmethod
    def cleanUpData(data, colRemove = None):
        
        # Remove unlabelled columns
        unlabelled_columns = [c for c in data.columns if c.split(' ')[0] in 'Unnamed:']
        data.drop(unlabelled_columns, axis = 1, inplace = True)
        
        extra_col = ['Sample', 'Package Loss (%)', 'Bandwidth (Kbps)']
        
        for col in extra_col:
            if col in data.columns:
                data.drop(col, axis = 1, inplace = True)
                
        # Remove all rows with nan
        data = data.dropna() 
        #print(data)
        
        # For each row, for some columns
        # average certain parameters per unit time of call duration
        columns_to_average = ['PL','PLC','PR','OS','OR','PS']
        for row in range(data.shape[0]):
            for column in columns_to_average:
                col = data.columns.get_loc(column)
                data.iat[row,col] = data.iat[row,col] / data['DU'][row]
                
        col_to_remove = ['ST','PS', 'OS', 'OR', 'DU', 'R', 'TR']
        
        # Decide which parameters are not likely to be predictive of call quality
        if colRemove is not None:
            if colRemove == 8:
                col_to_remove = ['PS', 'OS', 'OR', 'DU', 'MT', 'EN', 'ST', 'WTM', 'WTA', 'WTX', 'TR', 'A', 'R', 'T', 'M', 'C'] # 8
            elif colRemove == 7:
                col_to_remove = ['PS', 'OS', 'OR', 'DU', 'MT', 'EN', 'ST', 'WTM', 'WTA', 'WTX', 'TR', 'A', 'R', 'T', 'M', 'C', 'BRM'] # 7
            elif colRemove == 11:
                col_to_remove = ['PS', 'OS', 'OR', 'DU', 'MT', 'EN', 'ST', 'TR', 'A', 'R', 'T', 'M'] # 11
            elif colRemove == 16:
                col_to_remove = ['PS', 'OS', 'OR', 'DU', 'MT', 'EN', 'ST', 'R', 'TR'] # 16
            else:
                col_to_remove = ['ST','PS', 'OS', 'OR', 'DU', 'R', 'TR']

        for col in col_to_remove:
            if col in data.columns:
                data.drop(col, axis = 1, inplace = True)

        # Remove columns where all values are the same
        data = DataHandler.remove_column_if_all_rows_same(data, [0, -1, 124])
        return data

    @staticmethod
    def predictionData(pathToPredictFile, pathToTrainingData, skiprowsTest = 0, skiprowsTrain = 0, colRemove = None, targetName = 'MOS', normalizeData = True):
        
        predictionData = pd.read_csv(pathToPredictFile, skiprows = skiprowsTest)
        cleanedData = DataHandler.cleanUpData(predictionData, colRemove = colRemove)
        
        # Extract values
        feature_columns = [c for c in cleanedData.columns.values if c not in [targetName]]
        target_columns = [c for c in cleanedData.columns.values if c in [targetName]]
        #print('Features: ', feature_columns)
        hasTarget = True
        # If there are no labels default to last column
        if len(target_columns) == 0:
            hasTarget = False    
            #target_columns = cleanedData.columns.values[-1]
                
        # Extract relevant columns from data frame as numpy arrays
        X = cleanedData[feature_columns]
        
        X_return = None
        y_return = np.array([])
        
        if normalizeData:
            X_scaler, y_scaler = DataHandler.getXYTransformer(pathToTrainingData, skiprows = skiprowsTrain, colRemove=colRemove)
            
            X_return = X_scaler.transform(X)
            
            if hasTarget:
                y = cleanedData[target_columns].values
                # Reshape target data into a 2d array [[N x 1]]
                y = y.reshape(-1, 1)
                y_return = y_scaler.transform(y)
            
        else:
            X_return = X
            if hasTarget:
                y = cleanedData[target_columns].values
                # Reshape target data into a 2d array [[N x 1]]
                y_return = y.reshape(-1, 1)
                
        return X_return, y_return
    
    @staticmethod
    def getXYTransformer(pathToData, skiprows = 0, colRemove = None):
        # Get normalization transformer
        data = DataHandler()
        data.loadData(pathToData, skiprows = skiprows, colRemove = colRemove)
        _, _, _, _ = data.getSplitData(normalizeData = True, testSplit = 0.0)
        
        
        X_scaler = data.X_scaler
        y_scaler = data.y_scaler
        
        return X_scaler, y_scaler
        
    
    def saveNormParamsAsJson(self, filename = 'normalization-parameters.json'):
        
        # Ensure normalization parameters have been set
        if self.X_scaler is None or self.y_scaler is None:
            self.getSplitData(normalizeData = True)

        params = self.getNormParams(verbose = False)
        features = self.feature_columns
        targets = self.target_columns
        
        dataDict = dict()
        
        for f in range(len(features)):    
            dataDict[features[f]] = {'Mean' : params[0][0][f], 'SD' : params[0][1][f]}
        
        for t in range(len(targets)):
            dataDict[targets[t]] = {'Mean' : params[1][0][t], 'SD' : params[1][1][t]}
        
        jsonData = json.dumps(dataDict)

        with open(filename,'w') as json_file:
            json_file.write(jsonData)
    
    def predict(self, regressionOperator, X = None, y = None):
        """ 
        Returns an array of predictions for the particular
        operator if it supports the predict method and possibly
        the cross validated score array. If y_ground_truth is not None.
        
        """
        if X is None:
            X = self.X
            y = self.y
            
        return regressionOperator.predict(X, y)

        
    def getSplitData(self, X = None, y = None, validationData = False, normalizeData = False, testSplit = 0.2,
                     validationSplit = 0.25, testState = None, valState = None):
            
        assert(testSplit <= 1.0 and testSplit >= 0)
        assert(validationSplit <= 1.0 and validationSplit >= 0)
        
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        
        if normalizeData:
            # Normalize data to have a std of 1 and mean of 0
            self.X_scaler =  StandardScaler()
            X = self.X_scaler.fit_transform(X)
            
            self.y_scaler = StandardScaler()
            y = self.y_scaler.fit_transform(y)
        
        
        if validationData == True:
            # Generate the split: X_train 50 %, test 16.66 %, validation-1 16.66 %, validation-2 16.66 %
            X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size = 0.5, random_state = testState)
            
            X_remain, X_test, y_remain, y_test = train_test_split(X_remain, y_remain, test_size = 1/3.0, random_state = valState)
            X_remain, X_val_1, y_remain, y_val_1 = train_test_split(X_remain, y_remain, test_size = 1/2.0, random_state = valState)
            X_val_2, _, y_val_2,_ = train_test_split(X_remain, y_remain, test_size = 0, random_state = valState)

            return X_train, y_train, X_test, y_test, X_val_1, y_val_1, X_val_2, y_val_2
        
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSplit, random_state = testState)

        return X_train, X_test, y_train, y_test

    def getNormParams(self, verbose = True):
        """ Returns the mean and the std scaling for the X and y data """
        if verbose:
            print("X mean: {}]\nX scale: {}\ny mean: {}\ny scale: {}".format(self.X_scaler.mean_, self.X_scaler.scale_, self.y_scaler.mean_, self.y_scaler.scale_))
        return [self.X_scaler.mean_, self.X_scaler.scale_], [self.y_scaler.mean_, self.y_scaler.scale_]
    

    def getTruncatedDataLessTraining(self, percentageOfData, validationData = False, normalizeData = False):
        
        assert(percentageOfData >= 0 and percentageOfData <= 100), "Data used must be a value between 0 and 100 %"
        
        if validationData:
            X_train, X_test, y_train, y_test, X_val, y_val = self.getSplitData(testState = None, valState = None, validationData = validationData, normalizeData = normalizeData)
            limit = int(X_train.shape[0] * percentageOfData/100.0)
            return X_train[1:limit], X_test, y_train[1:limit], y_test, X_val, y_val
        
        X_train, X_test, y_train, y_test = self.getSplitData(testState = None, valState = None, validationData = validationData)
        limit = int(X_train.shape[0] * percentageOfData/100.0)
        return X_train[1:limit], X_test, y_train[1:limit], y_test
        
    
    def getTruncatedData(self, percentageOfData, validationData = False, normalizeData = False):
        
        assert(percentageOfData >= 0 and percentageOfData <= 100), "Data used must be a value between 0 and 100 %"
        
        # Extract relevant columns from data frame as numpy arrays
        limit = int(self.data.shape[0] * percentageOfData/100.0)
        data_trunc = self.data[1:limit]
        
        X = data_trunc[self.feature_columns].values
        y = data_trunc[self.target_columns].values
        
        # Reshape data into a 2d array (N x 1)
        y = y.reshape(-1, 1)
        
        # Split data into test and train sets
        return self.getSplitData(X, y, testState = None, valState = None, validationData = validationData, normalizeData = normalizeData)
    
    
    @staticmethod
    def remove_column_if_all_rows_same(data, constraintValue = None):
        """ Removes a column from a DF if all rows along a column contains the same values """
        
        columns = (c for c in data.columns)
        
        for c in columns:
            if len(set(data[c])) == 1:
                if constraintValue is not None:
                    if data[c].values[0] in constraintValue:
                        data.drop(c, axis = 1, inplace = True)
                else:
                    data.drop(c, axis = 1, inplace = True)
        return data

    def plot_loss_band_3d(self):
        return
        #import matplotlib.pyplot as plt
        #from mpl_toolkits.mplot3d import Axes3D
        #
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(data_loss_band['Package Loss (%)'].values, data_loss_band['Bandwidth (Kbps)'].values, y)
        #ax.set_xticks([0,5,10,15,20])
        #ax.set_yticks([25,35,45,55,65,75])
        #
        #ax.set_xlabel('Package Loss (%)')
        #ax.set_ylabel('Bandwidth (Kbps)')
        #ax.set_zlabel('MOS', rotation = 90)
        
        #ax.view_init(0, 0) # Bandwidth
        #ax.view_init(0, 90) # Package Loss    
        #X = data[feature_columns].values
        #y = data[target_columns].values
        
    def pearson(self):
        """ Calculates the pearson correlation coefficient for each feature with respect to the y column (MOS) """
        E = lambda x : np.sum(x)/len(x)
        sigma = lambda x : np.sqrt(E(x**2)-E(x)**2)
        pearson = lambda x,y : (E(x*y)-E(x)*E(y))/(sigma(x)*sigma(y))
        
        result = dict()
        for col in [columns for columns in self.data.columns if columns != 'MOS']:
            p = pearson(self.data[col], np.ravel(self.y))
            result[col] = p
        
        return result
