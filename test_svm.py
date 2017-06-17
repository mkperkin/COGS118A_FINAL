from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.sparse import csr_matrix
import os
import sys
import numpy as np

import classifiers
from SENTIMENT import helpers

X, y = helpers.load_data()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                            random_state=0)


print('Xtrain:' + str(X_train.shape))
print(X_train.head())

print('ytrain:' + str(y_train.shape))
print(y_train.head())

print('Xtest:' + str(X_test.shape))
print(X_test.head())
print('ytest:' + str(y_test.shape))
print(y_test.head())


param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}] #}
                  #  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


clf, params = classifiers.simple_svm(csr_matrix(X_train), y_train, param_grid)


print("\n-- Best Parameters:")
for k, v in params.items():
    print("parameter: {:<20s} setting: {}".format(k, v))


print("\n\n-- Testing best parameters [Random]...")
clf = SVC(**params)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()),
                                          end="\n\n" )
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))



