#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle 
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, BayesianRidge, HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from datetime import datetime

class regression_models:
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
		#y_test = self.train_LassoCV(data)
		#y_test = self.train_rigdeCV(data)
		#y_test = self.train_elasticNetCV(data)
		#y_test = self.train_BayesianRidge(data)
		#y_test = self.train_HuberRegressor(data)
		#y_test = self.train_SVM(data)
		#y_test = self.train_krrl_linear(data)
		self.save_prediction(y_test)


	def prepare_data(self):
		x_cols = [x for x in self.df_train.columns if x != 'precio' and x != 'id']
		y_train = np.log1p(self.df_train['precio'])

		x_train = self.df_train.loc[:,x_cols]
		self.x_test = self.df_test.loc[:,x_cols]
		
		scaler = RobustScaler()
		x_train = scaler.fit_transform(x_train)
		self.x_test = scaler.transform(self.x_test)


		x_tr,x_val,y_tr,y_val = train_test_split(x_train,y_train,test_size=0.15,shuffle=True)

		return (x_tr,y_tr),(x_val,y_val)

	def train_LassoCV(self,data):
		train,validacion = data
		x_tr,y_tr = train
		x_val,y_val = validacion
		#print("El set de train tiene {} filas y {} columnas".format(x_tr.shape[0],x_tr.shape[1]))
		#print("El set de validacion tiene {} filas y {} columnas".format(x_val.shape[0],x_val.shape[1]))

		print('Start training LassoCV...')
		start_time = self.timer()

		Lasso = LassoCV(
			n_alphas=1000,
			cv=10,
			normalize=True,

			)
		Lasso.fit(x_tr,y_tr)
		print("The R2 is: {}".format(Lasso.score(x_tr,y_tr)))
		print("The alpha choose by CV is:{}".format(Lasso.alpha_))
		self.timer(start_time)

		print("Making prediction on validation data")
		y_val = np.expm1(y_val)
		y_val_pred = np.expm1(Lasso.predict(x_val))
		mae = mean_absolute_error(y_val,y_val_pred)
		print("El mean absolute error de es {}".format(mae))

		
		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/LassoCV.pkl','wb') as f:
			pickle.dump(Lasso, f)

		print('Making prediction and saving into a csv')
		y_test= Lasso.predict(self.x_test)

		return y_test

	def train_rigdeCV(self,data):
		train,validacion = data
		x_tr,y_tr = train
		x_val,y_val = validacion
		#print("El set de train tiene {} filas y {} columnas".format(x_tr.shape[0],x_tr.shape[1]))
		#print("El set de validacion tiene {} filas y {} columnas".format(x_val.shape[0],x_val.shape[1]))

		print('Start training Ridge...')
		start_time = self.timer()

		ridge = RidgeCV(
			normalize=True,
			alphas=np.arange(0.000001,0.0001,0.000001),
			cv=10
			)
		ridge.fit(x_tr,y_tr)
		print("The R2 is: {}".format(ridge.score(x_tr,y_tr)))
		print("The alpha choose by CV is:{}".format(ridge.alpha_))
		self.timer(start_time)

		print("Making prediction on validation data")
		y_val = np.expm1(y_val)
		y_val_pred = np.expm1(ridge.predict(x_val))
		mae = mean_absolute_error(y_val,y_val_pred)
		print("El mean absolute error de es {}".format(mae))

		
		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/RidgeCV.pkl','wb') as f:
			pickle.dump(ridge, f)

		print('Making prediction and saving into a csv')
		y_test= ridge.predict(self.x_test)

		return y_test

	def train_elasticNetCV(self,data):
		train,validacion = data
		x_tr,y_tr = train
		x_val,y_val = validacion
		#print("El set de train tiene {} filas y {} columnas".format(x_tr.shape[0],x_tr.shape[1]))
		#print("El set de validacion tiene {} filas y {} columnas".format(x_val.shape[0],x_val.shape[1]))

		print('Start training ElasticNetCV...')
		start_time = self.timer()

		enet = ElasticNetCV(
			normalize=True,
			n_alphas=2000,
			max_iter = 2000,
			cv=10
			)
		enet.fit(x_tr,y_tr)
		print("The R2 is: {}".format(enet.score(x_tr,y_tr)))
		print("The alpha choose by CV is:{}".format(enet.alpha_))
		self.timer(start_time)

		print("Making prediction on validation data")
		y_val = np.expm1(y_val)
		y_val_pred = np.expm1(enet.predict(x_val))
		mae = mean_absolute_error(y_val,y_val_pred)
		print("El mean absolute error de es {}".format(mae))

		
		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/enetCV.pkl','wb') as f:
			pickle.dump(enet, f)

		print('Making prediction and saving into a csv')
		y_test= enet.predict(self.x_test)

		return y_test

	def train_BayesianRidge(self,data):
		train,validacion = data
		x_tr,y_tr = train
		x_val,y_val = validacion
		#print("El set de train tiene {} filas y {} columnas".format(x_tr.shape[0],x_tr.shape[1]))
		#print("El set de validacion tiene {} filas y {} columnas".format(x_val.shape[0],x_val.shape[1]))

		print('Start training BayesianRidge...')
		start_time = self.timer()

		bayesr = BayesianRidge(
			normalize = True,
			n_iter = 1000
			)
		bayesr.fit(x_tr,y_tr)
		print("The R2 is: {}".format(bayesr.score(x_tr,y_tr)))
#		print("The alpha choose by CV is:{}".format(bayesr.alpha_))
		self.timer(start_time)

		print("Making prediction on validation data")
		y_val = np.expm1(y_val)
		y_val_pred = np.expm1(bayesr.predict(x_val))
		mae = mean_absolute_error(y_val,y_val_pred)
		print("El mean absolute error de es {}".format(mae))

		
		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/bayesrCV.pkl','wb') as f:
			pickle.dump(bayesr, f)

		print('Making prediction and saving into a csv')
		y_test= bayesr.predict(self.x_test)

		return y_test
	
	def train_HuberRegressor(self,data):
		train,validacion = data
		x_tr,y_tr = train
		x_val,y_val = validacion
		#print("El set de train tiene {} filas y {} columnas".format(x_tr.shape[0],x_tr.shape[1]))
		#print("El set de validacion tiene {} filas y {} columnas".format(x_val.shape[0],x_val.shape[1]))

		print('Start training HuberRegressor...')
		start_time = self.timer()

		hr = HuberRegressor(
			max_iter = 1000,
			epsilon = 1,
			alpha = 0.1
			)
		hr.fit(x_tr,y_tr)
		print("The R2 is: {}".format(hr.score(x_tr,y_tr)))
#		print("The alpha choose by CV is:{}".format(hr.alpha_))
		self.timer(start_time)

		print("Making prediction on validation data")
		y_val = np.expm1(y_val)
		y_val_pred = np.expm1(hr.predict(x_val))
		mae = mean_absolute_error(y_val,y_val_pred)
		print("El mean absolute error de es {}".format(mae))

		
		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/hrCV.pkl','wb') as f:
			pickle.dump(hr, f)

		print('Making prediction and saving into a csv')
		y_test= hr.predict(self.x_test)

		return y_test

	def train_SVM(self,data):
		train,validacion = data
		x_tr,y_tr = train
		x_val,y_val = validacion
		#print("El set de train tiene {} filas y {} columnas".format(x_tr.shape[0],x_tr.shape[1]))
		#print("El set de validacion tiene {} filas y {} columnas".format(x_val.shape[0],x_val.shape[1]))

		print('Start training LinearSVR...')
		start_time = self.timer()

		svr = LinearSVR(
			max_iter=1800
			)
		svr.fit(x_tr,y_tr)
		print("The R2 is: {}".format(svr.score(x_tr,y_tr)))
		self.timer(start_time)

		print("Making prediction on validation data")
		y_val = np.expm1(y_val)
		y_val_pred = np.expm1(svr.predict(x_val))
		mae = mean_absolute_error(y_val,y_val_pred)
		print("El mean absolute error de es {}".format(mae))

		
		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/svrCV.pkl','wb') as f:
			pickle.dump(svr, f)

		print('Making prediction and saving into a csv')
		y_test= svr.predict(self.x_test)

		return y_test

	def train_krrl_linear(self,data):
		train,validacion = data
		x_tr,y_tr = train
		x_val,y_val = validacion
		#print("El set de train tiene {} filas y {} columnas".format(x_tr.shape[0],x_tr.shape[1]))
		#print("El set de validacion tiene {} filas y {} columnas".format(x_val.shape[0],x_val.shape[1]))

		print('Start training KernerRidge with linear kernel...')
		start_time = self.timer()

		krrl = KernelRidge(
			alpha=1
			)
		krrl.fit(x_tr,y_tr)
		print("The R2 is: {}".format(krrl.score(x_tr,y_tr)))
#		print("The alpha choose by CV is:{}".format(krrl.alpha_))
		self.timer(start_time)

		print("Making prediction on validation data")
		y_val = np.expm1(y_val)
		y_val_pred = np.expm1(krrl.predict(x_val))
		mae = mean_absolute_error(y_val,y_val_pred)
		print("El mean absolute error de es {}".format(mae))

		
		print('Saving model into a pickle')
		try:
			os.mkdir('pickles')
		except:
			pass

		with open('pickles/krrlLinearK.pkl','wb') as f:
			pickle.dump(krrl, f)

		print('Making prediction and saving into a csv')
		y_test= krrl.predict(self.x_test)

		return y_test


	def save_prediction(self,y_test):
		final_pred = np.expm1(y_test)

		ids = self.df_test['id'].values

		submit = pd.DataFrame({'id':ids,'target':final_pred})
		submit.to_csv('submit-lassoCV.csv',index=False)
	
	def timer(self, start_time=None):
		if not start_time:
			start_time = datetime.now()
			return start_time
		elif start_time:
			thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
			tmin, tsec = divmod(temp_sec, 60)
			print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

if __name__ == '__main__':
	predict = regression_models()