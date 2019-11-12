# %matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.ma.bench import timer
from scipy.sparse.linalg import svds
import category_encoders as ce
from xgboost import XGBClassifier, XGBRegressor
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import pickle
import os

class Inicializacion():
    # Encoder = 0 #Binario
    # Encoder = 1 #One Hot Encoding
    def __init__(self, encoder=0):
        self.encoder = encoder
        # TODO: Modo verboso para informe, generando png de los gráfico de nulls, o bien escribiendo y poniendolo en
        #       la pantalla
        # TODO: Params como diccionario de parametros
        print("Inicializando dataframes")
        df = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')

        self.df_final = self.operaciones(df)
        self.df_final_test = self.operaciones(df_test)

        # Opcion de salida de CSV y modelos, para evitar perder el tiempo de computo
        self.df_final.to_csv('00-df_final.csv')
        self.df_final_test.to_csv('01-df_final_test.csv')

    def getDataframes(self):
        return self.df_final, self.df_final_test

    def operaciones(self, df):
        print("Comenzando operaciones")
        df = self.tratamiento_nulls(df)
        df = self.encoding(df)
        df = self.casteos(df)
        df = self.features_engineering(df)
        df = self.predict_nulls(df)
        # Drop nan que quedaron pos limpieza y tratamientos
        df = df.dropna()
        return df

    def casteos(self, df):
        print("\t Cast")
        # Casteamos los dtypes a los correctos.
        return df.astype({
            "piscina": 'int16',
            "usosmultiples": 'int16',
            "escuelascercanas": 'int16',
            "centroscomercialescercanos": 'int16',
            "gimnasio": 'int16',
            #    "antiguedad": 'int16',
            #    "habitaciones": 'int16',
            #    "banos": 'int16',
            #    'garages': 'int16',
            "metroscubiertos": 'int16',
            "metrostotales": 'int16',
            "fecha": np.datetime64
        })

    def tratamiento_nulls(self, df):
        print("\t Nulls")
        df = self.drop_cols(df)
        df = self.drop_nan(df)
        df = self.fill_metros(df)
        return df

    def drop_cols(self, df):
        print("\t\t Drop cols")
        # Latitud y Longitud son importantes pero la cantidad de nulls es muy grande
        return df.drop(columns=['lat', 'lng', 'titulo', 'descripcion', 'id', 'idzona', 'direccion'])  # , inplace=True)

    def drop_nan(self, df):
        print("\t\t Drop nan")
        return df.dropna(subset=['tipodepropiedad', 'provincia', 'ciudad'])  # , inplace=True)

    def fill_metros(self, df):
        print("\t\t Fill metros")
        print("\t\t\t Terreno")
        df1 = df[df.tipodepropiedad == 'Terreno'].fillna(0)
        df = df[df.tipodepropiedad != 'Terreno']
        df = pd.concat([df, df1])
        # Terreno != null
        cond1 = (df.tipodepropiedad == 'Terreno') & (df.metroscubiertos != 0)
        df.metroscubiertos = np.where(cond1, 0, df.metroscubiertos)

        print("\t\t\t Apartamento")
        cond = (df.tipodepropiedad == 'Apartamento') & (df.metrostotales.isnull())
        df.metrostotales = np.where(cond, df.metroscubiertos, df.metrostotales)

        print("\t\t\t Metros cubiertos null")
        # Metros totales = metros cubiertos SI es null
        df.metrostotales = np.where(df.metrostotales.isnull(), df.metroscubiertos, df.metrostotales)
        print("\t\t\t Metros totales null")
        # Metros cubiertos = metros totales SI es null
        df.metroscubiertos = np.where(df.metroscubiertos.isnull(), df.metrostotales, df.metroscubiertos)

        print("\t\t\t Terreno comercial")
        df1 = df[df.tipodepropiedad == 'Terreno comercial'].fillna(0)
        df = df[df.tipodepropiedad != 'Terreno comercial']
        return pd.concat([df, df1])

    def predict_nulls(self, df):
        print("\t\t Predict nulls")
        df = self.fill_xgboost(df, 'garages')
        df = self.fill_xgboost(df, 'banos')
        df = self.fill_xgboost(df, 'habitaciones')
        df = self.fill_xgboost(df, 'antiguedad', True)
        return df

    def fill_xgboost(self, df, feature, continua=False):
        print("\t\t\t fill with xgboost. Feature: ", {feature})
        # TODO: Format columnas según encoding usado
        # Columnas relevantes
        cols = ['antiguedad', 'habitaciones',
                'tipodepropiedad_0bc',
                'tipodepropiedad_1bc',
                'tipodepropiedad_2bc',
                'tipodepropiedad_3bc',
                'tipodepropiedad_4bc',
                'tipodepropiedad_5bc',
                'ciudad_0bc',
                'ciudad_1bc',
                'ciudad_2bc',
                'ciudad_3bc',
                'ciudad_4bc',
                'ciudad_5bc',
                'ciudad_6bc',
                'ciudad_7bc',
                'ciudad_8bc',
                'ciudad_9bc',
                'ciudad_10bc',
                'provincia_0bc',
                'provincia_1bc',
                'provincia_2bc',
                'provincia_3bc',
                'provincia_4bc',
                'provincia_5bc',
                'banos', 'metroscubiertos', 'metrostotales',
                'gimnasio', 'garages', 'usosmultiples', 'piscina', 'escuelascercanas',
                'centroscomercialescercanos']
        # Sin fecha y sin precio
        # Usamos todas la que queremos predecir
        cols_subset = [x for x in cols if x != feature]

        df_train = df.dropna()
        # df_train
        df_test = df.loc[df[feature].isnull() == True]
        df_test = (df_test.dropna(subset=cols_subset))

        # Separamos los datos.
        df_train_x = df_train.loc[:, cols_subset]
        df_train_y = df_train[feature]
        df_test_y = df_test[feature]

        df_test_x = df_test.dropna(subset=cols_subset)

        df_test_x = df_test_x.loc[:, cols_subset]
        df_test = pd.merge(df_test_x, df_test_y.to_frame(), how='inner', left_index=True, right_index=True)
        df_test_x = df_test_x.loc[:, cols_subset]

        # Modelo - XGBoost
        # Los parametros son fruta, estan puestos solo para hacer pruebas.
        params = {
            'min_child_weight': [5],
            'gamma': [0.5],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'max_depth': [4],
            'n_estimators': [50],
            'learning_rate': [0.001, 0.01, 0.03]}
        if continua:
            xgb = XGBRegressor()
            scoring = 'neg_mean_squared_error'
        else:
            xgb = XGBClassifier()
            scoring = 'accuracy'

        # TODO: Folds y param_comb como parametros.
        # TODO: Parametro que levante el último modelo generado y use eso?
        folds = 2
        param_comb = 1

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

        random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=scoring,
                                           n_jobs=-1,
                                           cv=skf.split(df_train_x, df_train_y), random_state=1001)

        # Here we go
        start_time = self.timer(None)  # timing starts from this point for "start_time" variable
        random_search.fit(df_train_x, df_train_y)
        df_test_x[feature + '_xgb'] = random_search.predict(df_test_x)

        # df = df con modelo aplicado.
        df = pd.merge(df, df_test_x[feature + '_xgb'].to_frame(), how='left', left_index=True, right_index=True)
        df[feature] = np.where((df[feature].isnull() == True), df[feature + '_xgb'], df[feature])
        df.drop(columns=[feature + '_xgb'], inplace=True)
        self.df_xgb = df
        self.timer(start_time)  # timing ends here for "start_time" variable

        # Resultados CV
        results = random_search.cv_results_
        results = pd.DataFrame(results)
        results.to_csv('models/01-cv_results_' + feature + '.csv')

        # Dictionary of best parameters
        best_pars = random_search.best_params_
        print(' Los mejores parametros son: ')
        print(best_pars)
        print('------------------------')
        print('Best score: ')
        print(random_search.best_score_)

        # Best XGB model that was found based on the metric score you specify
        best_model = random_search.best_estimator_
        # Save model
        pickle.dump(random_search.best_estimator_, open("models\\00-nulls-xgb_" + feature + ".pickle", "wb"))

        return df

    def encoding(self, df):
        print("\t Encoding")
        catlist = ['tipodepropiedad', 'ciudad', 'provincia']
        # 0 - Binario
        # 1 - One hot Encoding
        if self.encoder == 0:
            # Binary Encoding
            binary_enc = ce.BinaryEncoder()
            binary_encoded = binary_enc.fit_transform(df[catlist])
            df = df.join(binary_encoded.add_suffix('bc'))
        elif self.encoder == 1:
            # One hot Encoding
            one_hot_enc = ce.OneHotEncoder()
            one_hot_encoded = one_hot_enc.fit_transform(df[catlist])
            df = df.join(one_hot_encoded.add_suffix('oh'))
        # TODO: Otros encodings?

        return df.drop(columns=catlist)  # , inplace=True)

    def features_engineering(self, df):
        print("\t Features engineering")
        # Partir fecha
        print("\t\t Separar fecha")
        df.assign(
            day=df.fecha.dt.day,
            month=df.fecha.dt.month,
            year=df.fecha.dt.year)
        return df.drop(columns='fecha')  # , inplace=True)

    def timer(self, start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

    def mostrar_nulls(self, df,feature,save=False):
        nulls = pd.DataFrame((df.isnull().sum().sort_values() / len(df) * 100).round(2), columns=['porcentaje de NaN'])
        nulls.drop(nulls.loc[nulls.loc[:, 'porcentaje de NaN'] <= 0].index, inplace=True)
        plt.figure(figsize=(12, 8))
        ax = nulls['porcentaje de NaN'].plot.barh()
        ax.set_title('Porcentaje de valores nulos en cada columna', fontsize=20, y=1.02)
        ax.set_xlabel('Porcentaje del total %', fontsize=16)
        ax.set_ylabel('columnas', fontsize=16)
        ax.grid(axis='x')

        for y, x in enumerate(nulls['porcentaje de NaN']):
            ax.text(x, y, s=str(x) + '%', color='black', fontweight='bold', va='center')

        script_dir = os.path.dirname(__file__)
        plots_dir = os.path.join(script_dir, 'plots/')
        file_name = feature

        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)

        if save == True:
            plt.savefig(plots_dir+file_name+'.png')
        else:
            pass

        plt.show()


if __name__ == '__main__':
    Inicializacion.Inicilizacion(0)
