# %matplotlib inline
import os
import pickle
from datetime import datetime

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor


class Inicializacion():
    # Encoder = 0 #Binario
    # Encoder = 1 #One Hot Encoding
    paramsGenerales = {
        'verbose': False,  # mostrar imagenes post transformacion
        'guardarImagenes': False,  # Guarda imagenes post transformacion
        'encoder': 1,  # Encoder. 0 Binario, 1 One hot encoding
        'usarModelo': False,  # Si esta el modelo de pickle lo usa
        'dropNan': True,  # Si es test no droppeamos nans
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
        self.paramsGenerales['dropNan'] = False
        self.df_final_test = self.operaciones(df_test)

        # Opcion de salida de CSV y modelos, para evitar perder el tiempo de computo
        self.df_final.to_csv('00-df_final.csv')
        self.df_final_test.to_csv('01-df_final_test.csv')

    def getDataframes(self):
        return self.df_final, self.df_final_test

    def operaciones(self, df):
        print("Comenzando operaciones")
        if self.paramsGenerales['verbose']:
            print("Cantidad de registros: ", len(df))
        df = self.tratamiento_nulls(df)
        if self.paramsGenerales['verbose']:
            print("Cantidad de registros: ", len(df))
        df = self.encoding(df)
        if self.paramsGenerales['verbose']:
            print("Cantidad de registros: ", len(df))
        if self.paramsGenerales['dropNan']:
            print("\t drop nans in selected columns")
            df = self.drop_nan(df)

        # Descartamos las columnas que fueron encodeadas
        df = df.drop(columns=['tipodepropiedad', 'provincia', 'ciudad'])
        df = self.casteos(df)
        if self.paramsGenerales['verbose']:
            print("Cantidad de registros: ", len(df))
        df = self.features_engineering(df)
        if self.paramsGenerales['verbose']:
            print("Cantidad de registros: ", len(df))
        df = self.predict_nulls(df)
        if self.paramsGenerales['verbose']:
            print("Cantidad de registros: ", len(df))

        # Drop nan que quedaron pos limpieza y tratamientos
        if self.paramsGenerales['dropNan']:
            print("\t drop other nans")
            df = df.dropna()
            df = self.recast(df)
            # self.metric_selection(df)
            if self.paramsGenerales['verbose']:  # TODO: único verbose?
                print("Cantidad de registros: ", len(df))
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

        # TODO: Validar, si no existe el modelo, aunque quisiera usarlo no puede. Corregir eso.
        if self.paramsGenerales['usarModelo']:
            df_test_x = df.loc[:, cols_subset]

            random_search = pickle.load(open("models/00-nulls-xgb_" + feature + ".pickle", 'rb'))
            # result = loaded_model.score(X_test, Y_test)
            df_test_x[feature + '_xgb'] = random_search.predict(df_test_x)
            # print(result)
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
            folds = self.paramsGenerales['folds']
            param_comb = self.paramsGenerales['param_comb']

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

            if continua:
                xgb = XGBRegressor()
                scoring = self.paramsGenerales['scoring_regressor']
                cv=5
            else:
                xgb = XGBClassifier()
                scoring = self.paramsGenerales['scoring_cat']
                cv=skf.split(df_train_x, df_train_y)  # 'accuracy'  #


            random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=scoring,
                                               n_jobs=-1,
                                               cv=cv, random_state=1001)

            # Here we go
            random_search.fit(df_train_x, df_train_y)

            df_test_x[feature + '_xgb'] = random_search.predict(df_test_x)

        # df = df con modelo aplicado.
        df = pd.merge(df, df_test_x[feature + '_xgb'].to_frame(), how='left', left_index=True, right_index=True)
        df[feature] = np.where((df[feature].isnull()), df[feature + '_xgb'], df[feature])
        df.drop(columns=[feature + '_xgb'], inplace=True)
        # self.df_xgb = df
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
        # TODO: AUC Scoring?

        return df

    def features_engineering(self, df):
        print("\t Features engineering")
        # Partir fecha
        print("\t\t Separar fecha")
        df = df.assign(
            day=df.fecha.dt.day,
            month=df.fecha.dt.month,
            year=df.fecha.dt.year)
        df = df.drop(columns='fecha')
        return df

    def metric_selection(self, df):
        print("\t Metric selection")
        feature_cols = df.columns.drop('precio')  # TODO: Q es nuestro outcome? Precio?
        train, valid, _ = self.get_data_splits(df)

        print('\t\t f_classif')
        # Empezamos a contar el tiempo
        start_time = self.timer(None)  # timing starts from this point for "start_time" variable

        selector = SelectKBest(f_classif, k=3)

        X_new = selector.fit_transform(train[feature_cols], train['precio'])
        plt.bar(feature_cols, selector.scores_)
        plt.xlabel('Features')
        plt.xticks(rotation=90)
        plt.ylabel('Importancia')
        plt.title('Importancia Features con Univariate (f_classif)')
        plt.show()
        self.timer(start_time)  # timing ends here for "start_time" variable

        # print('\t\t chi2')
        # # Empezamos a contar el tiempo
        # start_time = self.timer(None)  # timing starts from this point for "start_time" variable

        # selector_chi2 = SelectKBest(chi2, k=3)
        # X_new = selector_chi2.fit_transform(train[feature_cols], train['precio'])
        # plt.bar(feature_cols, selector_chi2.scores_)
        # plt.xlabel('Features')
        # plt.ylabel('Importancia')
        # plt.title('Importancia Features con Univariate (chi2)')
        # plt.show()
        #
        # # Sin la primera
        # plt.bar(feature_cols[1:], selector_chi2.scores_[1:])
        # plt.xlabel('Features')
        # plt.xticks(rotation=90)
        # plt.ylabel('Importancia')
        # plt.title('Importancia Features con Univariate (chi2)')
        # plt.show()
        # self.timer(start_time)  # timing ends here for "start_time" variable

        print('\t\t mutual_info_classif')
        # Empezamos a contar el tiempo
        start_time = self.timer(None)  # timing starts from this point for "start_time" variable
        selector_mutual = SelectKBest(mutual_info_classif, k=3)

        X_new = selector_mutual.fit_transform(train[feature_cols], train['precio'])
        plt.bar(feature_cols, selector_mutual.scores_)
        plt.xlabel('Features')
        plt.xticks(rotation=90)
        plt.ylabel('Importancia')
        plt.title('Importancia Features con Univariate (mutual)')
        plt.show()
        self.timer(start_time)  # timing ends here for "start_time" variable

        print('\t\t L1 regularization (Lasso Regression)')
        # Empezamos a contar el tiempo
        start_time = self.timer(None)  # timing starts from this point for "start_time" variable
        train, valid, _ = self.get_data_splits(df)
        X, y = train[train.columns.drop("precio")], train['precio']

        # Set the regularization parameter C=1
        logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(X, y)
        plt.bar(feature_cols, logistic.coef_[0])
        plt.xlabel('Features')
        plt.xticks(rotation=90)
        plt.ylabel('Importancia')
        plt.title('Importancia Features con Lasso')
        plt.show()
        self.timer(start_time)  # timing ends here for "start_time" variable

        print('\t\t Random forest')
        # Empezamos a contar el tiempo
        start_time = self.timer(None)  # timing starts from this point for "start_time" variable
        X, y = train[train.columns.drop("precio")], train['precio']
        val_X, val_y = valid[valid.columns.drop("precio")], valid['precio']

        forest_model = RandomForestClassifier(random_state=1)
        forest_model.fit(X, y)
        preds = forest_model.predict(val_X)

        plt.bar(feature_cols, forest_model.feature_importances_)
        plt.xlabel('Features')
        plt.xticks(rotation=90)
        plt.ylabel('Importancia')
        plt.title('Importancia Features con RF')
        plt.show()
        self.timer(start_time)  # timing ends here for "start_time" variable

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

    def get_data_splits(self, df, valid_fraction=0.1):
        # valid_fraction = 0.1
        # print(type(df))
        # print(df.shape())
        # print(df.head(5))

        valid_size = int(len(df) * valid_fraction)

        train = df[:-valid_size * 2]
        # valid size == test size, last two sections of the data
        valid = df[-valid_size * 2:-valid_size]
        test = df[-valid_size:]

        return train, valid, test

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
