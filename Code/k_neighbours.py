#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# K Nearest Neighbour Regression

from sklearn import neighbors
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

n_neighbors = 1

X_train, X_test, y_train, y_test = data.getSplitData(validationData = False, normalizeData = True)

knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

y_pred_inv = y_scaler.inverse_transform(y_pred)
y_test_inv = y_scaler.inverse_transform(y_test)

plt.figure()
plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Predicated')
plt.ylabel('Actual')
plt.axhline(y = 5, color = 'g', linestyle = '--', label='MOS = 5')
plt.axhline(y = 1, color = 'g', linestyle = '--', label='MOS = 1')


y = lambda x: x*1
x = np.linspace(np.min(y_pred_inv), np.max(y_pred_inv), len(y_pred_inv)/3)
y = y(x)

plt.plot(x,y, color = 'r', label='y=x')
plt.legend()

print("MSE: %.2f" %(metrics.mean_squared_error(y_pred, y_test)))
print("MAE: %.2f" %(metrics.mean_absolute_error(y_pred, y_test)))
print("Explained Variance: %.2f" %(metrics.explained_variance_score(y_pred, y_test)))
print("R^2: %.2f" %(metrics.r2_score(y_pred, y_test)))
