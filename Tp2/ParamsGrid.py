import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor


class ParamsGrid:
    metodos = {
        'RandomForest': (RandomForestRegressor(), {
            'bootstrap': [True, False],  # method for sampling data points (with or without replacement)
            'max_depth': np.append(None, np.arange(2, 20, 3)),  # max number of levels in each decision tree
            'max_features': ['auto', 'sqrt'],  # max number of features considered for splitting a node
            'min_samples_leaf': np.arange(10, 30, 5),  # min number of data points allowed in a leaf node
            'min_samples_split': np.arange(2, 20, 3),
            'n_estimators': np.arange(1200, 1800, 200) # number of trees in the foreset
        }),
        'xgb': (XGBClassifier(), {
            'min_child_weight': np.arange(1, 5, 0.1),
            'gamma': np.arange(0, 3, 0.2),
            'colsample_bytree': [0.6],
            'colsample_bynode': [0.6],
            'colsample_bylevel': [0.6],
            'max_depth': np.arange(3, 20, 3),
            'n_estimators': np.arange(500, 2000, 100),
            'learning_rate': [0.001, 0.01, 0.1, 0.4, 0.7, 0.9, 1, 1.2],
            'reg_alpha': np.arange(0.7, 3, 0.1)
        }),
        'knn': (KNeighborsRegressor(), {
            'n_neighbors': np.arange(15, 34, 2),
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        }),
        'gbr': (GradientBoostingRegressor(),
                {'n_estimators': [250],
                 'max_depth': [3],
                 'learning_rate': [.1, .01, .001],
                 'min_samples_leaf': [9],
                 'min_samples_split': [9]})
    }


class singleModels:
    models = {
        'gbr': GradientBoostingRegressor(loss='quantile', alpha=0.95,
                                         n_estimators=250, max_depth=3,
                                         learning_rate=.1, min_samples_leaf=9,
                                         min_samples_split=9)
    }
