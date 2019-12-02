#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import xgboost as xgb
from datetime import datetime

class Regressor:
	def __init__(self):
		print('Loading data...')
		self.df_train = pd.read_csv('00-df_final_ok.csv')
		self.df_train = self.df_train.dropna()
		self.df_test = pd.read_csv('01-df_final_test_ok.csv')
        
		for x in ['zonas_exclusivas', 'refaccion', 'lujo', 'vigilancia', 'country', 'es_Casa', 'es_Apartamento', 'es_Casa_en_condominio', 'es_Terreno', 'es_Local_Comercial']:
			self.df_train[x] = self.df_train[x].astype(np.int16)
			self.df_test[x] = self.df_test[x].astype(np.int16)

		print('El set de train tiene {} filas y {} columnas'.format(self.df_train.shape[0],self.df_train.shape[1]))
		print('El set de test tiene {} filas y {} columnas'.format(self.df_test.shape[0],self.df_test.shape[1]))

		self.do_pipeline()
	
	def do_pipeline(self):
		xgb_train,xgb_eval = self.prepare_data()
		y_test = self.train_model_nocv(xgb_train, xgb_eval)
		self.save_prediction(y_test,'XGBoost')

	def prepare_data(self):
		y_train = np.log(self.df_train['precio'])

		x_cols = [x for x in self.df_train.columns if x != 'precio' and x != 'id']
		x_train = self.df_train.loc[:,x_cols]
		x_test = self.df_test.loc[:,x_cols]

		self.x_test_dmatrix = xgb.DMatrix(x_test)



		x_tr,x_eval,y_tr,y_val = train_test_split(x_train,y_train,test_size=0.15,shuffle=True)
		xgb_train = xgb.DMatrix(x_tr,y_tr)
		xgb_eval = xgb.DMatrix(x_eval,y_val)

		return  xgb_train,xgb_eval	

	def accuracy_plot(self,y_val_pred,save=False):
		predictions = [round(value) for value in y_val_pred]
		self.mae = mean_absolute_error(np.exp(self.eval_set[1][1]),np.exp(y_val_pred))
		print(self.mae)

		results = self.model.evals_result()
		epochs = len(results['validation_0']['mae'])
		x_axis = range(0, epochs)

		fig, ax = plt.subplots()
		ax.plot(x_axis, results['validation_0']['mae'], label='Train')
		ax.plot(x_axis, results['validation_1']['mae'], label='Test')
		plt.ylabel('MAE')
		plt.xlabel('cantidad de árboles')
		plt.title('Variación del mean absolute error')
		plt.show()

		if save==True:
			plt.savefig('plots/xgboost-mae.png')

	def train_model_nocv(self,xgb_train,xgb_eval):
		param = {
		'eval_metric':'mae',
		'learning_rate':0.07,
		'gamma':0,
		'colsample_bytree':0.6,
		'colsample_bynode':0.6,
		'colsample_bylevel':0.6,
		'max_depth':9,
		'min_child_weight':0.8,
		'reg_alpha':1.8,
		'random_state':42,
		}
		

		print('Start training...')
		start_time = self.timer()

		watchlist=[(xgb_eval,'eval'),(xgb_train,'train')]

		booster = xgb.train(
				param,
				xgb_train,
				1200,
				watchlist,
				verbose_eval=200
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

		with open('pickles/xgboost.pkl','wb') as f:
			pickle.dump(booster, f)

		print('Making prediction and saving into a csv')
		y_test= booster.predict(self.x_test_dmatrix)

		return y_test


	def randomizedGrid_and_CV(self):
		param = {
		'n_estimators':[1200],
		'learning_rate':[0.07],
		'gamma':[0],
		'colsample_bytree':[0.6],
		'colsample_bynode':[0.6],
		'colsample_bylevel':[0.6],
		'max_depth':[9],
		'min_child_weight':[0.8],
		'reg_alpha':[1.8],
		'random_state':[42],
		'verbosity':[0]
		}

		rgs = RandomizedSearchCV(
			estimator=XGBRegressor(),
			param_distributions=param,
			n_iter=1,
			random_state=42,
			n_jobs=-1,
			cv=5
			)
		rgs.fit(self.x_train,self.y_train)
		print(rgs.best_score_)
		self.y_pred = rgs.predict(self.x_test)


	def save_prediction(self,y_test,model):
		final_pred = np.expm1(y_test)
		ids = self.df_test['id'].values
		try:
			os.mkdir('predictions')
		except:
			pass

		submit = pd.DataFrame({'id':ids,'target':final_pred})
		submit.to_csv('predictions/submit-'+model+'.csv',index=False)
	
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
