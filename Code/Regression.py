#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score

import numpy as np

class LinearRegression(object):
    
    def __init__(self):
        self.bitmask = None
        self.coef = None
        self.intercept = None
        self.results = dict()
        self.reg = None
        self.type = 'Linear Regression'
        self.trained = False
        
    def fit(self, X, y, verbose = True, cv = 10):
        
        assert (self.trained), "Model must first be trained to obtain the optimal parameters"
    
        X_processed = self.processPredictionData(X)
        
        reg = linear_model.LinearRegression()
        reg.fit(X_processed, y)
        
        # Set the coefficient and intercept
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.reg = reg
        
        score = cross_val_score(estimator = reg,
                                X = X_processed,
                                y = y,
                                cv = cv,
                                verbose = 0,
                                scoring = 'neg_mean_squared_error')
        
        if verbose:
            print('Details of {} model fit. Evaluated with {}-fold CV:\nMSE:\t\t{:.4f}\nSD of MSE:\t{:.4f}\nRMSE:\t\t{:.4f}\nSD of RMSE:\t{:.4f}'.format(self.type, 
                  cv, (-1*score).mean(), 
                  (-1*score).std(), 
                  np.sqrt((-1*score).mean()),
                  np.sqrt((-1*score).std())
                  ))
        
    def train(self, features, X_train, X_test, y_train, y_test, n_jobs = -1, cv = 10, verbose = False, startIndex = 1):
        # We must loop over the data 2^cols times such that we perform linear regression
        # for every possible sub-selection of parameters to find the set of parameters
        # that provides the best accuracy'
        
        endIndex = 2**(features)

        mean_score = np.inf
        
        for i in range(startIndex, endIndex):    
            
            # Get the binary mask
            binary_vector = self.getBinaryMask(i, features)
            # Transform the feature set based on the binary mask
            X_train_reduced = self.applyBinaryMaskTransform(binary_vector, X_train)
                
            # Perform regression fitting with k-fold cross validation
            reg = linear_model.LinearRegression()
            scores = cross_val_score(estimator = reg, 
                                     X = X_train_reduced, 
                                     y = y_train, 
                                     cv = cv, 
                                     n_jobs = n_jobs,  # change to -1 for all CPUs
                                     verbose = 0,
                                     scoring = 'neg_mean_squared_error')
            
            mean = -1*scores.mean()
            #std = scores.std()
            
            if (mean < mean_score):    # Not >= prefer simpler models
                self.bitmask = binary_vector
                
                if verbose:
                    print("New best fit:", self.bitmask, "MSE: ", mean)
                    
                mean_score = mean
                
            #if (i % 10**2 == 0):
                #print(i)
            #print(mean)
            #print(std)
        
        # End of loop
        
        assert(sum(self.bitmask) > 0)
        
        if verbose:
            print("Parameter mask that provided the best fit:", self.bitmask)
        
        X_test_reduced = self.applyBinaryMaskTransform(self.bitmask, X_test)
        X_train_reduced = self.applyBinaryMaskTransform(self.bitmask, X_train)
        
        reg.fit(X_train_reduced, y_train)
        #reg.fit(X, y)
        
        # Set the coefficient and intercept
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.reg = reg
        # Perform the inverse normalization on the coefficients and intercepts
        #coef = coef / (applyBinaryMaskTransform(best_fit, X_scaler.scale_)) * y_scaler.scale_[0]
        #intercept = intercept / (X_scaler.scale_) * y_scaler.scale_[0]

        y_pred_test = reg.predict(X_test_reduced)
        y_pred_train = reg.predict(X_train_reduced)
        
        self.storeResults(y_pred_test, y_pred_train, y_test, y_train)
        self.trained = True
        
        if verbose:
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_test))  
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_test))  
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))  
            print('R^2 Score: ', metrics.r2_score(y_test, y_pred_test))
      
    def predict(self, X_data, y_ground_truth, cv = 10,
                scoring = 'neg_mean_squared_error', n_jobs = 1):
        """ 
        Returns an array of predictions based on the input
        and an array of the cross validated scores, based on
        the selected scoring method.
        
        The number of parameters must match the expected parameters of the
        function that was fit; given by the bitmask.
        
        """
        
        X_data = self.processPredictionData(X_data)
        print(np.shape(X_data))
        if y_ground_truth is not None:
            scores = cross_val_score(estimator = self.reg, 
                             X = X_data, 
                             y = y_ground_truth, 
                             cv = cv, 
                             n_jobs = n_jobs,  # change to -1 for all CPUs
                             verbose = 0,
                             scoring = scoring)
            return self.reg.predict(X_data), scores
        
        return self.reg.predict(X_data)
    
    def processPredictionData(self, X_data):
        return self.applyBinaryMaskTransform(self.bitmask, X_data)

        
    def storeResults(self, y_pred_test, y_pred_train, y_test, y_train):
        
        types = ['train', 'test']
        pred = [y_pred_train, y_pred_test]
        actual = [y_train, y_test]
        for type_ in types:
            self.results[type_] = dict()
        for i in range(len(pred)):
            mae = metrics.mean_absolute_error(actual[i], pred[i])
            mse = metrics.mean_squared_error(actual[i], pred[i])
            rmse = np.sqrt(metrics.mean_squared_error(actual[i], pred[i]))
            r2 = metrics.r2_score(actual[i], pred[i])
            
            self.results[types[i]]['mae'] = mae
            self.results[types[i]]['mse'] =  mse
            self.results[types[i]]['rmse'] = rmse
            self.results[types[i]]['r2'] = r2
        
    def function(self, columnNames = ['X_'], featureStartIndex = None, 
                 CoefficientsOfOne = True, CoefSensitivity = False, printFunction = True):
    
        # Used to determine whether a feature is selected, based on the value
        # for its corresponding coefficient from the linear regression
        # Used to determine whether a coefficient is close to 1
        # and whether it should be shown or not depending on the state
        # of CoefficientsOfOne
        one_sensitivity = 10**(-8)
        coef_sensitivity = 10**(-100) 
        
        if CoefSensitivity:
            coef_sensitivity = 10**(-8) 

        feature_indicies = self.getIndicesGreaterThanValue(np.expand_dims(self.bitmask, axis = 0), 0)
        
        coef_indicies = self.getIndicesGreaterThanValue(self.coef, coef_sensitivity)
        
        function = []
        function.append(str(self.intercept[0]))
            
        # Gets the coefficient, exponent and index of each feature
        # and adds it to a list
        for index in coef_indicies:      
            
            component = ''
                
            # Display coefficient if not close to 1 unless that it allowed
            if (CoefficientsOfOne == True or 
                (CoefficientsOfOne == False and 
                 (abs(self.coef[0][index]) > 1 + one_sensitivity and 
                  abs(self.coef[0][index]) > 1 - one_sensitivity))):
                
                component += str(self.coef[0][index]) + '*'
                    
            component += '(' + str(columnNames[feature_indicies[index]%len(columnNames)])
            
            if featureStartIndex != None:
                component += str(featureStartIndex)

            else:
                component += str(feature_indicies[index] + 1)
                
            component += ')'
            
            function.append(component)
        
        if printFunction:
            print(' + '.join(function))
            
        return function
        
    
    def applyBinaryMaskTransform(self, mask, matrix):
        """ Removes all columns in matrix, where == 0 in mask"""
        
        col_to_delete = np.where(mask == 0)[0]
        
        if (sum(col_to_delete) > 0): 
            reduced_matrix = np.delete(matrix, col_to_delete, axis = 1)
            return reduced_matrix
        else:
            return matrix

    def getBinaryMask(self, integerValue, features):
        """ Returns base2 representation of number in vector format"""
        
        integerValue = abs(integerValue)
        
        # Create a binary bool vector
        binary = format(integerValue,"0{}b".format(features))
                
        # Set size of vector
        binary_vector = np.zeros(features)
        for j in range(features):
            binary_vector[j] = int(binary[j])
    
        return binary_vector
    
    def getIndicesGreaterThanValue(self, array, limit):
        #print(array)
        indices = np.argwhere(abs(array) > limit)
        #print(indices)
        
        x,y = zip(*indices)
        
        return np.array(y)

####################################################
#### End of Class
####################################################
    





class PolynomialRegression(LinearRegression):
    
    def __init__(self):
        super().__init__()
        self.poly = None
        self.degree = None
        self.type = 'Polynomial Regression'

    def train(self, features, X_train, X_test, y_train, y_test, 
            startIndex = 1, 
            exit_early = True, exit_early_error_limit = 4, 
            startDegree = 1, endDegree = 10,
            cv = 10, verbose = False):
        
        assert (startDegree > 0), "PolynomialFeatures(degree=n) only supports n > 0"
        
        endIndex = 2**(features)
        
        mean_score = np.inf
        error = np.inf
        
        # We must loop over the data 2^cols times such that we perform linear regression
        # for every possible sub-selection of parameters to find the set of parameters
        # that provides the best accuracy
        for i in range(startIndex, endIndex):  
            # Get the binary mask
            binary_vector = self.getBinaryMask(i, features)
                
            # Transform the feature set based on the binary mask
            X_train_reduced = self.applyBinaryMaskTransform(binary_vector, X_train)

            #############
            ## Inner loop
            #############
            
            # Need to loop over polynomial features from 1 to N
            # For every polynomial feature use every permutation of the features
            # to calculate the best set of parameters together with polynomial degree
            previous_error = 0
            error_increasing = 0
            for n in range(startDegree, endDegree):

                poly = PolynomialFeatures(degree=n)
                X_train_reduced_poly = poly.fit_transform(X_train_reduced)

                reg = linear_model.LinearRegression()
                #reg.fit(X_train_reduced_poly, y_train)
                
                #y_pred = reg.predict(X_val_reduced_poly)
                
                scores = cross_val_score(estimator = reg, 
                                     X = X_train_reduced_poly, 
                                     y = y_train, 
                                     cv = cv, 
                                     n_jobs = 1,  # change to -1 for all CPUs
                                     verbose = 0,
                                     scoring = 'neg_mean_squared_error')
                
#                pred_error = metrics.mean_squared_error(y_pred, y_val)
                pred_error = -1*scores.mean()
                
                if (exit_early == True):
                    if (previous_error < pred_error):
                        error_increasing += 1
                    #else:
                        #error_increasing -= 1
                        #error_increasing = 0
                        
                    if (error_increasing >= exit_early_error_limit):
                        break
                    
                if (pred_error < error):
                    self.poly = poly
                    self.bitmask = binary_vector
                    self.degree = n
                    if verbose:
                        print("(Degree) New potential best fit:", binary_vector, " Degree: ", n, ", MSE: ", pred_error)

                    if verbose:
                        X_test_reduced = self.applyBinaryMaskTransform(binary_vector, X_test)
                        X_test_reduced_poly = poly.fit_transform(X_test_reduced)
                        reg = linear_model.LinearRegression()
                        reg.fit(X_train_reduced_poly, y_train)
                        y_pred_test = reg.predict(X_test_reduced_poly)
                        print("Test MSE: {:.4f}".format(metrics.mean_squared_error(y_test, y_pred_test)))
                    error = pred_error
                previous_error = pred_error
               
            ##################
            ## Exit inner loop
            ##################
            
#
#            X_train_reduced_poly = self.poly.fit_transform(X_train_reduced)
#            X_val_reduced_poly = self.poly.fit_transform(X_val_reduced)
#            
#            reg = linear_model.LinearRegression()
#            
#            # Cross validate
#            scores = cross_val_score(estimator = reg, 
#                                     X = X_val_reduced_poly, 
#                                     y = y_val, 
#                                     cv = cv, 
#                                     n_jobs = 1,  # change to -1 for all CPUs
#                                     verbose = 0,
#                                     scoring = 'neg_mean_squared_error')
#            mean = -1*scores.mean()
#            #print("MSE: ", mean)
#            if (mean < mean_score): # Not <= prefer simpler models
#                self.bitmask = binary_vector
#                self.degree = degree_candidate
#                #degree_candidate = n
#                print("New best fit:", self.bitmask, " Degree: ", self.degree, ", MSE: ", mean)
#                mean_score = mean
        
        
        # End of loop
        
        assert (self.degree > 0 and sum(self.bitmask) > 0 and self.poly is not None)

        if verbose:
            print("Best parameter mask: ", self.bitmask)
            print("Polynomial degree: ", self.degree)
#        
        X_test_reduced = self.applyBinaryMaskTransform(self.bitmask, X_test)
        X_train_reduced = self.applyBinaryMaskTransform(self.bitmask, X_train)
        
        X_train_reduced_poly = self.poly.fit_transform(X_train_reduced)
        X_test_reduced_poly = self.poly.fit_transform(X_test_reduced)
        
        reg = linear_model.LinearRegression()
        reg.fit(X_train_reduced_poly, y_train)
        
        # Set the coefficient and intercept
        self.coef = reg.coef_
        self.intercept = reg.intercept_
        self.reg = reg
        
        # Make predictions
        y_pred_test = reg.predict(X_test_reduced_poly)
        y_pred_train = reg.predict(X_train_reduced_poly)
        
        self.storeResults(y_pred_test, y_pred_train, y_test, y_train)
        self.trained = True

        if verbose:
            print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred_test))  
            print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred_test))  
            print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
            print('R^2 Score: ', metrics.r2_score(y_test, y_pred_test))

    def processPredictionData(self, X_data):
        X_data = self.applyBinaryMaskTransform(self.bitmask, X_data)
        return self.poly.fit_transform(X_data)
        
    def function(self, featureStartIndex = None, PowersOfOne = False, CoefficientsOfOne = True, 
                 columnNames = ['X_'], printFunction = True, CoefSensitivity = False):
        
        """ 
        
        To create labels consistent with excel columns use:
            columnNames = List of column names
            featureStartIndex = Row in excel
            
        """
    
        # Used to determine whether a feature is selected, based on the value
        # for its corresponding coefficient from the linear regression
        coef_sensitivity = 10**(-100) 
        # Used to determine whether a coefficient is close to 1
        one_sensitivity = 10**(-12)
        
        coef_ = self.coef
        intercept_ = self.intercept
        
        if np.ndim(coef_) == 1:
            coef_= np.expand_dims(self.coef, axis = 0)
        if np.ndim(intercept_) == 0:
            intercept_= np.expand_dims(self.intercept, axis = 0)
            
        if CoefSensitivity:
            coef_sensitivity = 10**(-12) 
            
        coef_indicies = self.getIndicesGreaterThanValue(self.coef, coef_sensitivity)
    
        feature_indicies = self.getIndicesGreaterThanValue(np.expand_dims(self.bitmask, axis = 0), 0)
    
        powers = self.poly.powers_   
        #print(powers)
        #print(self.coef)
        #print(self.intercept)
        
        function = []
        
        function.append(str(intercept_[0]))
                    
        # Gets the coefficient, exponent and index of each feature
        # and adds it to a list
        for index in coef_indicies:
            
            # Get index where poly.powers_ is non-zero
            ind = self.getIndicesGreaterThanValue(np.expand_dims(powers[index], axis = 0), 0)
            
            component = ''
            for exponent in ind:
                
                # feature_indicies: the index of the feature found in the best_fit mask
                # coef: the coefficient corresponding to that feature
                # powers: the exponent for that feature
                
                # Display coefficient is not close to 1 unless that it allowed
                if (CoefficientsOfOne == True or 
                    (CoefficientsOfOne == False and 
                     (abs(coef_[0][index]) > 1 + one_sensitivity and 
                      abs(coef_[0][index]) > 1 - one_sensitivity))):
                    
                    # If we have a product of features e.g. 3 * X_3 * X_1
                    # the coefficient is only printed once
                    if (ind[0] == exponent):
                        component += str(coef_[0][index]) + '*'
                    
                component += '('
                # uses % len(...) to support any length up to exponent
                component += str(columnNames[feature_indicies[exponent]%len(columnNames)])
                if featureStartIndex != None:
                    component += str(featureStartIndex)

                else:
                    component += str(feature_indicies[exponent] + 1)

                # Only display exponent of 1 if it is allowed
                if (PowersOfOne == True or (PowersOfOne == False and powers[index][exponent] > 1)):
                    component += ')^' + str(powers[index][exponent])
                else:
                    component += ')'
                    
                if exponent != ind[-1]:
                    component += ' * '
            
            function.append(component)
            
        if printFunction:
            print(' + '.join(function))
            
        return function
    
    
####################################################
#### End of Class
####################################################
