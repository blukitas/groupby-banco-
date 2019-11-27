import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from xgboost import plot_importance


class Regressor:
	def __init__(self):
		#train_path = open('pickle','rb')
		#test_path = open('path','rb')

		#self.df_train = pickle.load(train_path)
		#self.df_test = pickle.load(test_path)
		self.df_train = pd.read_csv('00-df_final.csv')
		self.df_test = pd.read_csv('01-df_final_test.csv')
		self.prepare_data()
		self.train_model_nocv()

	def prepare_data(self):
		x_cols = [x for x in self.df_train.columns if x != 'precio' and x != 'id']
		self.y_train = np.log(self.df_train['precio'])

		self.x_train = self.df_train.loc[:,x_cols]
		self.x_test = self.df_test.loc[:,x_cols]
		

	def accuracy_plot(self,y_val_pred,save=False):
		predictions = [round(value) for value in y_val_pred]
		self.acc = accuracy_score(self.eval_set[1][1],y_val_pred)
		print(acc)

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

	def train_model_nocv(self):
		x_tr,x_val,y_tr,y_val = train_test_split(self.x_train,self.y_train,shuffle=True,test_size=0.15)
		self.eval_set = [(x_tr, y_tr), (x_val, y_val)]
		self.model = XGBRegressor(
			verbosity= 0,
			reg_alpha= 1.8,
			random_state= 42,
			n_estimators= 1200,
			min_child_weight= 0.8,
			max_depth= 9,
			learning_rate=0.07,
			gamma= 0.0,
			colsample_bytree= 0.6,
			colsample_bynode= 0.6,
			colsample_bylevel= 0.6
			)
		self.model.fit(x_tr,y_tr,eval_set=self.eval_set,eval_metric='error',verbose=0)
		y_val_pred = self.model.predict(x_val)

		self.accuracy_plot(y_val_pred)
		plot_importance(self.model,max_num_features=10)

		self.y_test = self.model.predict(self.x_test)

	def save_prediction(self,y_test):
		ids = self.df_test['id'].values
		submit = pd.DataFrame({'id':ids,'precio':self.y_test})
		submit.to_csv('submit{}.csv'.format(self.acc),index=False)

if __name__ == '__main__':
    predict = Regressor()
