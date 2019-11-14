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
    paramsGenerales = {
        'verbose': True,  # mostrar imagenes post transformacion
        'guardarImagenes': False,  # Guarda imagenes post transformacion
        'encoder': 0,  # Encoder. 0 Binario, 1 One hot encoding
        'usarModelo': False,  # Si esta el modelo de pickle lo usa
        'esTest': False,  # Si es test no droppeamos nans
        # Para xgboost
        'min_child_weight': [5],
        'gamma': [0.5],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'max_depth': [4],
        'n_estimators': [50],
        'learning_rate': [0.01],
        'scoring_regressor': 'neg_mean_squared_error',
        'scoring_cat': 'accuracy',
        'folds': 2,
        'param_comb': 1,
    }

    def __init__(self, params=[]):
        if params != []:
            self.paramsGenerales = params

        print("Inicializando dataframes")
        df = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')

        self.df_final = self.operaciones(df)

        self.paramsGenerales['usarModelo'] = True
        self.df_final_test = self.operaciones(df_test, dropNans=False)

        # Opcion de salida de CSV y modelos, para evitar perder el tiempo de computo
        self.df_final.to_csv('00-df_final.csv')
        self.df_final_test.to_csv('01-df_final_test.csv')

    def getDataframes(self):
        return self.df_final, self.df_final_test

    def operaciones(self, df, dropNans=True):
        print("Comenzando operaciones")
        print(len(df))
        df = self.tratamiento_nulls(df, dropNans=dropNans)
        print(len(df))
        df = self.encoding(df)
        print(len(df))
        if dropNans:
            print("\t drop nans in selected columns")
            df = self.drop_nan(df)
        # Descartamos las columnas que fueron encodeadas
        df = df.drop(columns=['tipodepropiedad', 'provincia', 'ciudad'])
        df = self.casteos(df)
        print(len(df))
        df = self.features_engineering(df)
        print(len(df))
        df = self.predict_nulls(df)
        print(len(df))
        # Drop nan que quedaron pos limpieza y tratamientos
        if dropNans:
            print("\t drop other nans")
            df = df.dropna()
        df = self.recast(df)
        print(len(df))
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

    def tratamiento_nulls(self, df, dropNans=True):
        print("\t Nulls")
        df = self.drop_cols(df)
        df = self.fill_metros(df)
        return df

    def drop_cols(self, df):
        print("\t\t Drop cols")
        return df.drop(columns=['lat', 'lng', 'titulo', 'descripcion', 'idzona', 'direccion'])  # , inplace=True)

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
        if self.paramsGenerales['verbose']:
            self.mostrar_nulls(df, 'garages')
        df = self.fill_xgboost(df, 'banos')
        if self.paramsGenerales['verbose']:
            self.mostrar_nulls(df, 'banos')
        df = self.fill_xgboost(df, 'habitaciones')
        if self.paramsGenerales['verbose']:
            self.mostrar_nulls(df, 'habitaciones')
        df = self.fill_xgboost(df, 'antiguedad', continua=True)
        if self.paramsGenerales['verbose']:
            self.mostrar_nulls(df, 'antiguedad')
        return df

    def fill_xgboost(self, df, feature, continua=False):
        print("\t\t\t fill with xgboost. Feature: ", {feature})
        # Columnas relevantes (Sin precio)
        cols = [x for x in df.columns if x != 'precio']

        # Todas - la que queremos predecir
        cols_subset = [x for x in cols if x != feature]

        # Empezamos a contar el tiempo
        start_time = self.timer(None)  # timing starts from this point for "start_time" variable

        if self.paramsGenerales['usarModelo']:
            df_test_x = df.loc[:, cols_subset]

            random_search = pickle.load(open("models/00-nulls-xgb_" + feature + ".pickle", 'rb'))
            #result = loaded_model.score(X_test, Y_test)
            df_test_x[feature + '_xgb'] = random_search.predict(df_test_x)
            #print(result)
        else:

            df_train = df.dropna()
            df_train.drop(columns=['id'])
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
                'min_child_weight': self.paramsGenerales['min_child_weight'],
                'gamma': self.paramsGenerales['gamma'],
                'subsample': self.paramsGenerales['subsample'],
                'colsample_bytree': self.paramsGenerales['colsample_bytree'],
                'max_depth': self.paramsGenerales['max_depth'],
                'n_estimators': self.paramsGenerales['n_estimators'],
                'learning_rate': self.paramsGenerales['learning_rate']
            }
            if continua:
                xgb = XGBRegressor()
                scoring = self.paramsGenerales['scoring_regressor']
            else:
                xgb = XGBClassifier()
                scoring = self.paramsGenerales['scoring_cat']  # 'accuracy'  #

            folds = self.paramsGenerales['folds']
            param_comb = self.paramsGenerales['param_comb']

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

            random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=scoring,
                                               n_jobs=-1,
                                               cv=skf.split(df_train_x, df_train_y), random_state=1001)

            # Here we go
            random_search.fit(df_train_x, df_train_y)

            df_test_x[feature + '_xgb'] = random_search.predict(df_test_x)

        # df = df con modelo aplicado.
        df = pd.merge(df, df_test_x[feature + '_xgb'].to_frame(), how='left', left_index=True, right_index=True)
        df[feature] = np.where((df[feature].isnull()), df[feature + '_xgb'], df[feature])
        df.drop(columns=[feature + '_xgb'], inplace=True)
        #self.df_xgb = df
        self.timer(start_time)  # timing ends here for "start_time" variable

        if not self.paramsGenerales['usarModelo']:
            # Creamos la carpeta si no existe

            script_dir = os.path.dirname(__file__)
            models_dir = os.path.join(script_dir, 'models/')
            file_name = feature

            if not os.path.isdir(models_dir):
                os.makedirs(models_dir)

            # Resultados CV
            results = random_search.cv_results_
            results = pd.DataFrame(results)
            results.to_csv('models/01-cv_results_' + feature + '.csv')

            # Dictionary of best parameters
            best_pars = random_search.best_params_
            print('\t\t\t\t Los mejores parametros son: ')
            print('\t\t\t\t', best_pars)
            print('\t\t\t\t ------------------------')
            print('\t\t\t\t Best score: ')
            print('\t\t\t\t', random_search.best_score_)
            print('\t\t\t\t ------------------------')

            # Best XGB model that was found based on the metric score you specify
            best_model = random_search.best_estimator_
            # Save model
            pickle.dump(random_search.best_estimator_, open("models/00-nulls-xgb_" + feature + ".pickle", "wb"))

        return df

    def encoding(self, df):
        print("\t Encoding")
        catlist = ['tipodepropiedad', 'ciudad', 'provincia']
        # 0 - Binario
        # 1 - One hot Encoding
        if self.paramsGenerales['encoder'] == 0:
            # Binary Encoding
            binary_enc = ce.BinaryEncoder()
            binary_encoded = binary_enc.fit_transform(df[catlist])
            df = df.join(binary_encoded.add_suffix('bc'))
        elif self.paramsGenerales['encoder'] == 1:
            # One hot Encoding
            one_hot_enc = ce.OneHotEncoder()
            one_hot_encoded = one_hot_enc.fit_transform(df[catlist])
            df = df.join(one_hot_encoded.add_suffix('oh'))
        # TODO: Otros encodings?

        return df

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
            print('\t\t\t Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

    def mostrar_nulls(self, df, feature):
        nulls = pd.DataFrame((df.isnull().sum().sort_values() / len(df) * 100).round(2), columns=['porcentaje de NaN'])
        if (nulls.sum() == 0).bool():
            print("No hay nulls")
        else:
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

            if self.paramsGenerales['guardarImagenes']:
                plt.savefig(plots_dir + file_name + '.png')

    def recast(self, df):
        print("\t Recast final")
        columns = []
        for x in df.columns:
            columns.append(x)

        try:
            columns.remove('precio')
        except:
            pass

        for x in range(len(columns)):
            dtype = df[columns[x]].dtype.type
            if dtype == np.int64 or dtype == np.float64:
                df = df.astype({columns[x]: np.int16})
        return df


if __name__ == '__main__':
    Inicializacion.Inicilizacion(0)
