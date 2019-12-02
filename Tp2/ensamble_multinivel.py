import pandas as pd
import numpy as np
import pickle 
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, BayesianRidge, HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
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
		self.df_train = pd.read_csv('00-df_final_ok.csv')
		self.df_test = pd.read_csv('01-df_final_test_ok.csv')
		self.y_tain = self.df_train['precio']
		self.df_train = self.df_train.drop(columns=['id','precio'])
		print(self.df_train.shape)

		self.pred = self.load_models()

		self.create_df_tier2()
	
	def load_models(self):
		lasso = pickle.load(open('pickles/LassoCV.pkl','rb'))
		lasso_pred = lasso.predict(self.df_train)
		
		lgb = pickle.load(open('pickles/lightgbm.pkl','rb'))
		lgb_pred = lgb.predict(self.df_train)
		
		bgr = pickle.load(open('pickles/bayesrCV.pkl','rb'))
		bgr_pred = bgr.predict(self.df_train)
		
		catboost = pickle.load(open('pickles/catboost.pkl','rb'))
		catboost_pred = catboost.predict(self.df_train)
		
		enet = pickle.load(open('pickles/enetCV.pkl','rb'))
		enet_pred = enet.predict(self.df_train)

		#etr = pickle.load(open('pickles/extr.pkl','rb'))
		#etr_pred = etr.predict(self.df_train)
		
		hr = pickle.load(open('pickles/hrCV.pkl','rb'))
		hr_pred = hr.predict(self.df_train)

		ridge = pickle.load(open('pickles/RidgeCV.pkl','rb'))
		ridge_pred = ridge.predict(self.df_train)

		svr = pickle.load(open('pickles/svrCV.pkl','rb'))
		svr_pred = svr.predict(self.df_train)

		baggingr = pickle.load(open('pickles/bg.pkl','rb'))
		baggingr_pred = baggingr.predict(self.df_train)

		xgb = pickle.load(open('pickles/xgboost.pkl','rb'))
		xgb_prd = xgb.predict(self.df_train)

		return [lasso_pred,lgb_pred, bgr_pred, catboost_pred, enet_pred, hr_pred, ridge_pred, svr_pred, baggingr_pred]

	def create_df_tier2(self):
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
		xgb = xgboost.XGBRegressor()
		xgb.fit(tier1,self.y_tain)
		y_pred_xgb = xgb.predict(self.df_train)

		lgb = lightgbm.LGBMRegressor()
		lgb.fit(tier1,self.y_tain)
		y_pred_lgm = lgb.predict(self.df_train)

		ctb = catboost.CatBoostRegressor()
		ctb.fit(tier1,self.y_tain)
		y_pred_cat = ctb.predict(self.df_train)


		columnas = ['y_pred_xgb','y_pred_lgm','y_pred_cat']
		pred = [y_pred_xgb,y_pred_lgm,y_pred_cat]

		tier2 = pd.DataFrame(columns=columnas)


		for columnas, pred in zip(columnas,pred):
			tier2[columnas] = pred

		return tier2


			
	def train_metaregressors_tier2(self,tier2):
		pass
