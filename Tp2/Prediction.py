from datetime import datetime
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from hypopt import GridSearch
import numpy as np
from Inicializacion import *
import math


class Prediction:

    def __init__(self, data, model, prefix, param_grid=[]):
        self.train_df, self.test_df  = data
        self.model = model
        self.param_grid = param_grid
        self.prefix = prefix + datetime.now().strftime('%m-%d-%H:%M')
        self.X = self.train_df.loc[:,  self.train_df.columns != 'precio']
        self.y = self.train_df['precio'].values
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, 
            self.y, test_size=0.1, random_state=1)
    
    def manualGridSearch(self):
        best_score = math.inf
        for g in self.param_grid:
            print(g)
            self.model.set_params(**g)
            self.model.fit(self.X_train, self.y_train)
            score = mean_absolute_error(self.model.predict(self.X_val), self.y_val)
            print(score)
            # save if best
            if score < best_score:
                self.best_score = score
                self.best_grid = g
        
    def gridSearchTrain(self):
        print('Training...')
        self.gscv = GridSearchCV(self.model, self.param_grid, scoring='neg_mean_absolute_error', verbose=10)
        self.gscv.fit(self.X_train, self.y_train)
        self.best_params = self.gscv.best_params_
        self.score = self.gscv.best_score_
        self.predicted = self.gscv.predict(self.test_df)
        print(self.best_params)
        print(self.score)
    
    def HypOptTrain(self):
        print('Training...')
        self.opt = GridSearch(model = self.model, param_grid = self.param_grid)
        self.opt.fit(self.X_train, self.y_train, self.X_val, self.y_val, scoring='neg_mean_squared_error')
        self.best_params = self.opt.best_params_
        self.score = self.opt.score(X_val, y_val)
        self.predicted = self.opt.predict(self.test_df)
        print(self.best_params)
        print(self.score)

    def train(self):
        print('Training...')
        self.model.fit(self.X_train, self.y_train)
        self.score = mean_absolute_error(self.model.predict(self.X_val), self.y_val)
        print(self.score) 
        self.predicted = self.model.predict(self.test_df)
    
    def crossValidation(self, cv=5):
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='neg_mean_absolute_error') #print each cv score (accuracy) and average them
        self.score = np.mean(cv_scores)
        print(self.score)

    def save(self):
        if self.param_grid==[]:
            with open('{}.model'.format(self.prefix),'wb') as f:
                pickle.dump(self.model, f)        
        else:       
            with open('{}.model'.format(self.prefix),'wb') as f:
                pickle.dump(self.gscv, f)

    def submit(self):
        self.test_ids = pd.read_csv('data/test.csv')['id']
        answer = pd.DataFrame(list(zip(self.test_ids, self.predicted)), columns =['id', 'target'])
        answer.to_csv('{}-{}.csv'.format(self.prefix, int(round(self.score))), sep=',', index=False)

