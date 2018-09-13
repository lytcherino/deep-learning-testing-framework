#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################
# Decision Tree
################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Classification

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)

score = cross_val_score(classifier, X_train, y_train, cv = 10)

print("CV MSE: %.2f\n" % (score.mean()))

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy: %.2f\nF1: %.2f" % (accuracy_score(y_pred, y_test), metrics.f1_score(y_pred, y_test)))

# Export as DOT file
# Using (graphviz): dot -Tsvg tree.dot -o tree.svg the tree can be viewed
tree.export_graphviz(classifier, out_file='tree.dot')


# Regression

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
prediction_error = np.array([])
training_error = np.array([])
max_depth = 50
for i in range(1,max_depth):
    classifier = DecisionTreeRegressor(criterion = 'mse', max_depth = i)
    classifier = classifier.fit(X_train, y_train)
    
    y_pred_test = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    
    pred_error_test = metrics.mean_squared_error(y_pred_test, y_test)
    pred_error_train = metrics.mean_squared_error(y_pred_train, y_train)
    
    prediction_error = np.append(prediction_error, pred_error_test)
    training_error = np.append(training_error, pred_error_train)

    print ('Max Depth: ', i, '(Test) Prediction Error: ', pred_error_test)
    print ('Max Depth: ', i, '(Training) Prediction Error: ', pred_error_train)

    
plt.figure()
plt.plot(np.arange(1,max_depth,1), prediction_error, label = 'Test')
plt.plot(np.arange(1,max_depth,1), training_error, label = 'Train')
plt.ylabel('MSE')
plt.xlabel('Max Depth')
plt.legend()




