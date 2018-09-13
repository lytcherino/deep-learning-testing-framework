#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

class DataAnalyser(object):

        def __init__(self, regressionOperator, feature_columns):
            self.mse_array_train = None
            self.mse_array_test = None
            self.training_data_size = None
            self.mse_array_train_dev = None
            self.mse_array_test_dev = None
            self.parameters_selected = None
            self.number_parameters_dev = None
            self.number_parameters_count = None
            self.reg = regressionOperator
            self.feature_columns = feature_columns
            
            self.histogram = None

        def calculateParameterFrequencies(self):

            # Get the average frequency of each feature for each training set size
            self.histogram = dict()
            for j in range(len(self.parameters_selected)):
                for i in range(len(self.training_data_size)):
                    for key, params in self.parameters_selected[j][i].items():
                        for param in params:
                            if param not in self.histogram.keys():
                                self.histogram[param] = np.zeros(len(self.training_data_size))
                            self.histogram[param][i] += 1
            
            # Average
            for key,value in self.histogram.items():
                self.histogram[key] = value / len(self.parameters_selected)
   
        def plotParameterProbabilities(self, axis = None):
            plt.figure()
            if axis is not None:
                plt.axis(axis)
                
            for key, value in self.histogram.items():
                color = ['black','red','green','blue','pink','purple','cyan', 'violet','lime','indigo','brown','gray']
                plt.scatter(self.training_data_size, value, label = key, color = color[self.feature_columns.index(key)])
                plt.plot(self.training_data_size, value, color = color[self.feature_columns.index(key)])
                plt.title('Parameter Pick Probability')
                plt.xlabel('Training Data Used')
                plt.ylabel('Chance the parameter is choosen (%)')
                plt.legend()
                
            return self

        def plotParameterCount(self, axis = None):
            
            plt.figure()
            if axis is not None:
                plt.axis(axis)
                
            plt.errorbar(self.training_data_size, self.number_parameters_count, yerr=self.number_parameters_dev, ms = 7.0, color = 'blue', mfc = 'cyan', fmt = '.', label = 'Parameter Count')
            plt.xlabel('Training Data Used')
            for i in range(int(np.min(self.number_parameters_count)),int(np.max(self.number_parameters_count)) + 1):
                plt.axhline(y = i, color = 'black')
            plt.legend()
            
            plt.figure()
            plt.plot(self.training_data_size, self.number_parameters_dev)
            plt.scatter(self.training_data_size, self.number_parameters_dev)
            plt.ylabel('Parameter Count Standard Deviation')
            plt.xlabel('Training Data Used')
            
            return self
            
        def plotMse(self, axis = None):
            
            plt.figure()
            if axis is not None:
                plt.axis(axis)
            
            plt.errorbar(self.training_data_size, self.mse_array_test, yerr=self.mse_array_test_dev, ms = 7.0, color = 'blue', mfc='cyan', fmt = '.', label = 'MSE Test')
            plt.plot(self.training_data_size, self.mse_array_test, color = 'black')
            
            plt.plot(self.training_data_size, self.mse_array_train, color = 'red')
            plt.errorbar(self.training_data_size, self.mse_array_train, yerr=self.mse_array_train_dev, ms = 7.0, color ='green', mfc='yellow', fmt = '.', label = 'MSE Train')
            
            title = '{} - MSE for Train & Test Data'.format(self.reg.type)
            plt.title(title)
            plt.xlabel('Training Data Used')
            plt.ylabel('Units of MSE')
            plt.legend()
            
            return self
        
        def plotMseStd(self, axis = None):
        
            plt.figure()
            if axis is not None:
                plt.axis(axis)
                
            title = '{} - STD of MSE, 10 fold CV'.format(self.reg.type)
            plt.title(title)
            plt.plot(self.training_data_size, self.mse_array_test_dev, label='STD of MSE Test', color = 'blue')
            plt.scatter(self.training_data_size, self.mse_array_test_dev, color = 'blue')
            
            plt.plot(self.training_data_size, self.mse_array_train_dev, label='STD of MSE Train', color = 'green')
            plt.scatter(self.training_data_size, self.mse_array_train_dev, color = 'green')
            
            plt.xlabel('Training Data Used')
            plt.ylabel('Units of MSE')
            plt.legend()
            
            return self
    
        def run(self, dataHandler, startPercentage = 10, endPercentage = 100, step = 1, repeats = 10, cv = 10, verbose = False):
            
            elements = (((endPercentage+1) - startPercentage) // step)
            self.mse_array_train = np.zeros(elements)
            self.mse_array_test = np.zeros(elements)
            self.training_data_size = np.zeros(elements)
            self.mse_array_train_dev = np.zeros(elements)
            self.mse_array_test_dev = np.zeros(elements)
            self.parameters_selected = np.array([np.array([dict() for i in range(elements)]) for i in range(repeats)])

            self.number_parameters_dev = np.zeros(elements)
            self.number_parameters_count = np.zeros(elements)
            
            for i in range(startPercentage, endPercentage + 1, step):
                print(i)
                train_results = np.array([])
                test_results = np.array([])
                number_parameters = np.array([])
                index = (i-startPercentage)//step
                for j in range(repeats):
                    #print(j)
                    X_train, X_test, y_train, y_test = dataHandler.getTruncatedDataLessTraining(i, validationData = False)
                    self.reg.fit(len(self.feature_columns), X_train, X_test, y_train, y_test, cv = cv, verbose = verbose)
                    train_results = np.append(train_results, self.reg.results['train']['mse'])
                    test_results = np.append(test_results, self.reg.results['test']['mse'])
                    self.training_data_size[index] = X_train.shape[0]
                    number_parameters = np.append(number_parameters, np.sum(self.reg.bitmask))
            
                    self.parameters_selected[j][index] = {i : np.array(self.feature_columns)[self.reg.bitmask == 1]}
                    
                self.mse_array_train_dev[index] = train_results.std()
                self.mse_array_test_dev[index] = test_results.std()
                self.mse_array_train[index] = train_results.mean()
                self.mse_array_test[index] = test_results.mean()
                self.number_parameters_count[index] = number_parameters.mean()
                self.number_parameters_dev[index] = number_parameters.std()
                
            # Post-process data
            self.calculateParameterFrequencies()
