import pandas as pd
import numpy as np
import pickle 
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, BayesianRidge, HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor
import xgboost
import catboost
import lightgbm
from datetime import datetime
import os

class Stacking:
	def __init__(self):
		self.paramsXGBoost = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

		self.df_train = pd.read_csv('00-df_final_ok.csv')
		self.df_train = self.df_train.sample(frac=0.0001)
		print(self.df_train.shape)

		self.df_test = pd.read_csv('01-df_final_test_ok.csv')
		self.y_tain = self.df_train['precio']
		self.df_train = self.df_train.drop(columns=['id','precio','prom_amb','construccion_density'])
		print(self.df_train.shape)

		self.pred = self.load_models()

		tier1 = self.create_df_tier1()
		tier2 = self.train_metaregressors_tier1(tier1)
		self.train_metaregressors_tier2(tier2)
	
	def load_models(self):
		adr = pickle.load(open('pickles/adr.pkl','rb'))
		print('Cargado AdaBoostRegressor. Empezando a predecir.')
		adr_pred = adr.predict(self.df_train)		

		extr = pickle.load(open('pickles/extr.pkl','rb'))
		print('Cargado Lasso. Empezando a predecir.')
		extr_pred = extr.predict(self.df_train)

		lasso = pickle.load(open('pickles/LassoCV.pkl','rb'))
		print('Cargado Lasso. Empezando a predecir.')
		lasso_pred = lasso.predict(self.df_train)
		
		lgb = pickle.load(open('pickles/lightgbm.pkl','rb'))
		print('Cargado LightGBM. Empezando a predecir.')
		lgb_pred = lgb.predict(self.df_train)
		
		bgr = pickle.load(open('pickles/bayesrCV.pkl','rb'))
		print('Cargado BayesianRidge. Empezando a predecir.')
		bgr_pred = bgr.predict(self.df_train)
		
		catboost = pickle.load(open('pickles/catboost.pkl','rb'))
		print('Cargado CatBoost. Empezando a predecir.')
		catboost_pred = catboost.predict(self.df_train)
	
		
		enet = pickle.load(open('pickles/enetCV.pkl','rb'))
		print('Cargado ElasticNetCV. Empezando a predecir.')
		enet_pred = enet.predict(self.df_train)
		
		hr = pickle.load(open('pickles/hrCV.pkl','rb'))
		print('Cargado HuberRegressor. Empezando a predecir.')
		hr_pred = hr.predict(self.df_train)

		ridge = pickle.load(open('pickles/RidgeCV.pkl','rb'))
		print('Cargado Ridge. Empezando a predecir.')

		ridge_pred = ridge.predict(self.df_train)

		svr = pickle.load(open('pickles/svrCV.pkl','rb'))
		print('Cargado Suport Vector Regressor. Empezando a predecir.')

		svr_pred = svr.predict(self.df_train)

		baggingr = pickle.load(open('pickles/bg.pkl','rb'))
		print('Cargado BaggingRegressor. Empezando a predecir.')
		baggingr_pred = baggingr.predict(self.df_train)

		xgb = pickle.load(open('pickles/xgboost.pkl','rb'))
		print('Cargado XGBoost. Empezando a predecir.')
		a = xgboost.DMatrix(self.df_train)

		xgb_prd = xgb.predict(a)

		return [adr_pred,extr_pred,lasso_pred,lgb_pred,bgr_pred,catboost_pred,xgb_prd,enet_pred,hr_pred,ridge_pred,svr_pred,baggingr_pred,xgb_prd]

	def create_df_tier1(self):
		columnas = ['lasso_pred',
			'lgb_pred',
			'bgr_pred',
			'catboost_pred',
			'enet_pred',
			'hr_pred',
			'ridge_pred',
			'svr_pred',
			'baggingr_pred']
		tier1 = pd.DataFrame(columns=columnas)
		for columnas,predicciones in zip(columnas, self.pred):
			tier1[columnas] = predicciones

		return tier1
		
	def train_metaregressors_tier1(self,tier1):

		paramsCatBoost = {}
		paramsLightGBM = {}

		xgb = xgboost.XGBRegressor()
		print('Entrenando XGBoost 50 combos y 5 folds')
		random_search = RandomizedSearchCV(xgb,
			param_distributions=self.paramsXGBoost,
			n_iter=50,
			scoring='neg_mean_absolute_error',
			n_jobs=-1,
			cv=5, random_state=1001)

		random_search.fit(self.df_train, self.y_tain)

		y_pred_xgb = random_search.predict(self.df_train)


		lgb = lightgbm.LGBMRegressor()
		
		print('Entrenando LightGBM 50 combos y 5 folds')

		random_search_lgb = RandomizedSearchCV(lgb,
			param_distributions=paramsLightGBM,
			n_iter=50,
			scoring='neg_mean_absolute_error',
			n_jobs=-1,
			cv=5, random_state=1001)

		random_search_lgb.fit(self.df_train, self.y_tain)

		y_pred_lgb = random_search_lgb.predict(self.df_train)		


		ctb = catboost.CatBoostRegressor()
		print('Entrenando CatBoost 50 combos y 5 folds')

		random_search_ctb = RandomizedSearchCV(ctb,
			param_distributions=paramsCatBoost,
			n_iter=50,
			scoring='neg_mean_absolute_error',
			n_jobs=-1,
			cv=5, random_state=1001)

		random_search_ctb.fit(self.df_train, self.y_tain)

		y_pred_ctb = random_search_ctb.predict(self.df_train)		

		columnas = ['y_pred_xgb','y_pred_lgm','y_pred_cat']
		pred = [y_pred_xgb,y_pred_lgm,y_pred_cat]

		tier2 = pd.DataFrame(columns=columnas)


		for columnas, pred in zip(columnas,pred):
			tier2[columnas] = pred

		return tier2
			
	def train_metaregressors_tier1(self,tier2):
		xgb = xgboost.XGBRegressor()

		random_search = RandomizedSearchCV(xgb,
			param_distributions=self.paramsXGBoost,
			n_iter=50,
			scoring='neg_mean_absolute_error',
			n_jobs=-1,
			cv=5, random_state=1001)

		random_search.fit(self.df_train, self.y_tain)

		y_pred_xgb = xgb.predict(self.df_test)

		final_pred = y_pred_xgb

		ids = self.df_test['id'].values
		try:
			os.mkdir('predictions')
		except:
			pass
		
		submit = pd.DataFrame({'id':ids,'target':final_pred})
		submit.to_csv('predictions/submit-ensamble_por_capas.csv',index=False)

if __name__ == '__main__':
	predict = Stacking()

		
