from __future__ import print_function


from classifiers import simple_bt


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import os
import sys
import numpy as np

from SENTIMENT import helpers

X, y = helpers.load_data()

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


print('Xtrain:' + str(X_train.shape))
print(X_train.head())

print('ytrain:' + str(y_train.shape))
print(y_train.head())

print('Xtest:' + str(X_test.shape))
print(X_test.head())
print('ytest:' + str(y_test.shape))
print(y_test.head())

#param_grid = {"n_estimators": [25, 50, 100, 150, 200, 250, 300],
 #             "learning_rate": [0.01, 0.05, 0.1, 1.0]}



param_grid = {"base_estimator__criterion" : ["gini","entropy"],
              "base_estimator__splitter" : ["best","random"],
              "base_estimator__max_depth" : [3, 4, 5, 6, 7, 8],
              "n_estimators": [300],
              "learning_rate": [0.01]}

clf, params= simple_bt(X_train,y_train, param_grid)


print("\n-- Best Parameters:")
for k, v in params.items():
    print("parameter: {:<20s} setting: {}".format(k, v))



print("\n\n-- Testing best parameters [Random]...")
dt = DecisionTreeClassifier(criterion='entropy',max_depth=7,splitter='random')
clf = AdaBoostClassifier(base_estimator=dt, n_estimators=40, learning_rate = 0.01)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()),
                                          end="\n\n" )

clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))



