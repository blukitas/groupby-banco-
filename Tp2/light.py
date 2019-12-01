#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import Dataset
from lightgbm import train
from lightgbm import cv
from datetime import datetime
import os


class Regressor:
	def __init__(self):
		print('Loading data...')
		self.df_train = pd.read_csv('00-df_final.csv')
		self.df_test = pd.read_csv('01-df_final_test.csv')
		

		print('El set de train tiene {} filas y {} columnas'.format(self.df_train.shape[0],self.df_train.shape[1]))
		print('El set de test tiene {} filas y {} columnas'.format(self.df_test.shape[0],self.df_test.shape[1]))
		self.do_pipeline()
	

	def do_pipeline(self):
		lgb_train,lgb_eval = self.prepare_data()
		y_test = self.train_model_nocv(lgb_train, lgb_eval)
		self.save_prediction(y_test)


	def prepare_data(self):
		x_cols = [x for x in self.df_train.columns if x != 'precio' and x != 'id']
		self.y_train = np.log1p(self.df_train['precio'])

		self.x_train = self.df_train.loc[:,x_cols]
		self.x_test = self.df_test.loc[:,x_cols]
		print(len(self.x_test))
		print()

		x_tr,x_eval,y_tr,y_val = train_test_split(self.x_train,self.y_train,test_size=0.15,shuffle=True)

		lgb_train = Dataset(x_tr,y_tr,free_raw_data=False)
		lgb_eval = Dataset(x_eval,y_val,free_raw_data=False)

		return lgb_train,lgb_eval


	def train_model_nocv(self,lgb_train,lgb_eval):
		params = {
#		'num_leaves':15,
		'boosting_type': 'gbdt',
		'metric': 'mean_absolute_error',
#		'num_boost_round':4000,
		'verbose': 0,
		'learning_rate':0.22,
#		'max_depth':10
		}
		


		print('Start training...')
		start_time = self.timer()

		booster = train(
				params=params,
				train_set=lgb_train,
				valid_sets=[lgb_eval,lgb_train],
				valid_names=['eval','train'],
				verbose_eval=10,
			#	learning_rates=lambda iter: 0.3 * (0.99 ** iter)
			)

		"""booster = cv(
			params=params,
			train_set=lgb_train,
			)"""

		self.timer(start_time)


		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/lightgbm.pkl','wb') as f:
			pickle.dump(booster, f)

		print('Making prediction and saving into a csv')
		y_test= booster.predict(self.x_test)

		return y_test

	def save_prediction(self,y_test):
		final_pred = np.expm1(y_test)
		ids = self.df_test['id'].values
		submit = pd.DataFrame({'id':ids,'target':final_pred})
		submit.to_csv('submit-lg.csv',index=False)
	
	def timer(self, start_time=None):
		if not start_time:
			start_time = datetime.now()
			return start_time
		elif start_time:
			thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
			tmin, tsec = divmod(temp_sec, 60)
			print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

if __name__ == '__main__':
	predict = Regressor()