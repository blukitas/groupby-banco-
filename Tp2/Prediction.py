from datetime import datetime
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from Inicializacion import *


class Prediction:

    def __init__(self, data, model, prefix, param_grid=[]):
        self.train_df, self.test_df  = data
        self.model = model
        self.param_grid = param_grid
        self.prefix = prefix + datetime.now().strftime('%m-%d-%H:%M')
        self.X = self.train_df.loc[:,  self.train_df.columns != 'precio']
        self.y = self.train_df['precio'].values
        #self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, 
        #	self.y, test_size=0.2, random_state=1)

    def gridSearchTrain(self):
        print('Training...')
        self.gscv = GridSearchCV(self.model, self.param_grid, scoring='neg_mean_absolute_error', cv=10, verbose=10)
        self.gscv.fit(self.X, self.y)
        self.best_params = self.gscv.best_params_
        self.score = self.gscv.best_score_
        self.predicted = self.gscv.predict(self.test_df)
        print(self.best_params)
        print(self.score)

    def train():
        self.model.fit(self.X, self.y)
        self.predicted = self.model.predict(self.test_df)

    def save(self):
        if self.param_grid=[]:
            with open('{}.model'.format(self.prefix),'wb') as f:
                pickle.dump(self.model, f)        
        else:       
            with open('{}.model'.format(self.prefix),'wb') as f:
                pickle.dump(self.gscv, f)

    def submit(self):
        self.test_ids = pd.read_csv('data/test.csv')['id']
        answer = pd.DataFrame(list(zip(self.test_ids, self.predicted)), columns =['id', 'predicted'])
        answer.to_csv('{}-{}.csv'.format(self.prefix, int(round(self.score))), sep=',', index=False)


if __name__ == '__main__':
    preprocesamiento = Inicializacion()
    data = (preprocesamiento.df_final, preprocesamiento.df_final_test) 
    distances = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    param_grid = {'n_neighbors': np.arange(15, 34, 2), 'metric': distances}
    model = Prediction(data, KNeighborsRegressor(), param_grid, 'knntest')
    model.train()
    model.save()
    model.submit()
