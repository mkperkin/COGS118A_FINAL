from __future__ import print_function

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import os
import sys
from time import time
from operator import itemgetter
import numpy as np


def simple_knn(X, y, param_grid):
    print("start knn...")
    clf = KNeighborsClassifier()
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()

    grid_search.fit(X, y)
        
    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)

    return grid_search, top_params
   

def simple_svm(X, y, param_grid):
    print("Beginning SVM...")
    clf = SVC()
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()

    grid_search.fit(X, y)
        
    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)

    return grid_search, top_params



def simple_bt(X,y, param_grid):
    dt = DecisionTreeClassifier() 
    clf = AdaBoostClassifier(base_estimator=dt)

    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,cv=5)
    start = time()

    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))
    sys.stdout.flush()
    top_params = report(grid_search.grid_scores_, 3)

    return grid_search, top_params





def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


    return top_scores[0].parameters

