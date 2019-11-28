#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from lightgbm import plot_importance
from lightgbm import LGBMRegressor
from lightgbm import plot_metric


class Regressor:
	def __init__(self):
		#train_path = open('pickle','rb')
		#test_path = open('path','rb')

		self.df_train,self.df_test = pd.read_pickle('dfsInicializados.pickle')

	
		self.prepare_data()
		self.train_model_nocv()

	def prepare_data(self):
		x_cols = [x for x in self.df_train.columns if x != 'precio' and x != 'id']
		self.y_train = np.log(self.df_train['precio'])

		self.x_train = self.df_train.loc[:,x_cols]
		self.x_test = self.df_test.loc[:,x_cols]	


	def train_model_nocv(self):
		x_tr,x_val,y_tr,y_val = train_test_split(self.x_train,self.y_train,shuffle=True,test_size=0.15)
		self.eval_set = [(x_tr, y_tr), (x_val, y_val)]
		self.model = LGBMRegressor(
			num_leaves=160,
			n_estimators=500)
		self.model.fit(x_tr,y_tr,eval_set=self.eval_set,eval_metric='mean_absolute_error',verbose=0)
		self.y_val_pred = self.model.predict(x_val)
		self.print_mae()

		#plot_importance(self.model,max_num_features=10)
		#plt.show()
	

		self.y_test = self.model.predict(self.x_test)
		self.save_prediction(self.y_test)

	def print_mae(self):
		mae = mean_absolute_error(np.exp(self.eval_set[1][1]),np.exp(self.y_val_pred))
		print(mae)



	def save_prediction(self,y_test):
		ids = pd.read_csv('data/test.csv')['id'].values
		final_pred = np.exp(self.y_test)
		submit = pd.DataFrame({'id':ids,'target':final_pred})
		submit.to_csv('submit-lg.csv',index=False)

if __name__ == '__main__':
	predict = Regressor()