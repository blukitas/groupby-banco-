import numpy as np


class ParamsGrid():
    params_xgboost = {
        'min_child_weight': [5],
        'gamma': [0.5],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'max_depth': [4],
        'n_estimators': [50],
        'learning_rate': [0.01]
    }

    param_random_forest = {
        'bootstrap': [True, False],  # method for sampling data points (with or without replacement)
        # 'criterion': 'mse',
        'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # max number of levels in each decision tree
        'max_features': ['auto', 'sqrt'],  # max number of features considered for splitting a node
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        'min_samples_leaf': [1, 2, 4, 10, 15],  # min number of data points allowed in a leaf node
        'min_samples_split': [2, 5, 10, 15, 20],  # min number of data points placed in a node before the node is split
        # 'min_weight_fraction_leaf': 0.0,
        # n_estimators = number of trees in the foreset
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        # 'n_jobs': 1,
        # 'oob_score': False,
        # 'random_state': 42,
        # 'verbose': 0,
        # 'warm_start': False
    }

    distances = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    params_knn = {
        'n_neighbors': np.arange(15, 34, 2),
        'metric': distances
    }
