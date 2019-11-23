import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBClassifier


class ParamsGrid:
    metodos = {
        (RandomForestRegressor(), {
            'bootstrap': [True, False],  # method for sampling data points (with or without replacement)
            # 'criterion': 'mse',
            'max_depth': np.append(None, np.arange(0, 200, 10)),  # max number of levels in each decision tree
            'max_features': ['auto', 'sqrt'],  # max number of features considered for splitting a node
            # 'max_leaf_nodes': None,
            # 'min_impurity_decrease': 0.0,
            # 'min_impurity_split': None,
            'min_samples_leaf': np.arange(2, 20, 2),  # min number of data points allowed in a leaf node
            'min_samples_split': np.arange(2, 20, 2),
            # min number of data points placed in a node before the node is split
            # 'min_weight_fraction_leaf': 0.0,
            # n_estimators = number of trees in the foreset
            'n_estimators': np.arange(200, 3000, 200),
            # 'n_jobs': 1,
            # 'oob_score': False,
            # 'random_state': 42,
            # 'verbose': 0,
            # 'warm_start': False
        }),
        (XGBClassifier(), {
            'min_child_weight': [5],
            'gamma': [1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'max_depth': np.arange(3, 20, 3),
            'n_estimators': np.arange(100, 2000, 100),
            'learning_rate': [0.01]
        }),
        (KNeighborsRegressor(), {
            'n_neighbors': np.arange(15, 34, 2),
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        })
    }
