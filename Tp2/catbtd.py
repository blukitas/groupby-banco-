#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from datetime import datetime
import os 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

class Catboost:
	def __init__(self):
		print('Loading data...')
		#self.df_train,self.df_test = pd.read_pickle('dfsInicializados.pickle')		
		self.df_train = pd.read_csv('00-df_final.csv')
		self.df_test = pd.read_csv('01-df_final_test.csv')


		
		print('El set de train tiene {} filas y {} columnas'.format(self.df_train.shape[0],self.df_train.shape[1]))
		print('El set de test tiene {} filas y {} columnas'.format(self.df_test.shape[0],self.df_test.shape[1]))

		self.do_pipeline()

	def do_pipeline(self):
		data = self.prepare_data()
		y_test = self.train(data)
		self.save_prediction(y_test)


	def prepare_data(self):
		x_cols = [x for x in self.df_train.columns if x != 'precio' and x != 'id']
		y_train = np.log1p(self.df_train['precio'])

		x_train = self.df_train.loc[:,x_cols]
		self.x_test = self.df_test.loc[:,x_cols]

		x_tr,x_val,y_tr,y_val = train_test_split(x_train,y_train,test_size=0.15,shuffle=True)

		return (x_tr,y_tr),(x_val,y_val)
		
	def train(self,data):
		train,validacion = data
		x_tr,y_tr = train
		x_val,y_val = validacion
		print("El set de train tiene {} filas y {} columnas".format(x_tr.shape[0],x_tr.shape[1]))
		print("El set de validacion tiene {} filas y {} columnas".format(x_val.shape[0],x_val.shape[1]))

		print("Start training..")
		start_time = self.timer()
		catb = CatBoostRegressor(
			eval_metric="MAE")
		catb.fit(
			x_tr,
			y_tr,
			eval_set=validacion,
			)

		self.timer(start_time)
		print("Making prediction on validation data")
		y_val = np.expm1(y_val)
		y_val_pred = np.expm1(catb.predict(x_val))
		mae = mean_absolute_error(y_val,y_val_pred)
		print("El mean absolute error de es {}".format(mae))

		
		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/catboost.pkl','wb') as f:
			pickle.dump(catb, f)

		print('Making prediction and saving into a csv')
		y_test= catb.predict(self.x_test)

		return y_test

		
	def save_prediction(self,y_test):
		final_pred = np.expm1(y_test)
		ids = self.df_test['id'].values
		submit = pd.DataFrame({'id':ids,'target':final_pred})
		submit.to_csv('submit-catboost.csv',index=False)
	
	def timer(self, start_time=None):
		if not start_time:
			start_time = datetime.now()
			return start_time
		elif start_time:
			thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
			tmin, tsec = divmod(temp_sec, 60)
			print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


if __name__ == "__main__":
	cat = Catboost()