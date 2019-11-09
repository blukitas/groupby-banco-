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


class Inicializacion():
    # Encoder = 0 #Binario
    # Encoder = 1 #One Hot Encoding
    def __init__(self, encoder=0):
        print("Inicializando dataframes")
        df = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')
        return self.operaciones(df, encoder), self.operaciones(df_test, encoder)

    def operaciones(self, df, encoder):
        df = self.casteos(self, df)
        df = self.tratamiento_nulls(self, df)
        df = self.encoding(self, df, encoder)
        df = self.features_engineering(self, df)
        df = self.predict_nulls(self, df)
        return df

    def timer(self, start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

    def casteos(self, df):
        # Casteamos los dtypes a los correctos.
        return df.astype({
            "piscina": 'int16',
            "usosmultiples": 'int16',
            "escuelascercanas": 'int16',
            "centroscomercialescercanos": 'int16',
            "gimnasio": 'int16',
            "antiguedad": 'int16',
            "habitaciones": 'int16',
            "banos": 'int16',
            "metroscubiertos": 'int16',
            "metrostotales": 'int16',
            'garages': 'int16',
            "fecha": np.datetime64
        })

    def tratamiento_nulls(self, df):
        df = self.drop_cols(self, df)
        df = self.drop_nan(self, df)
        df = self.fill_metros(self, df)
        return df

    def drop_cols(self, df):
        # Latitud y Longitud son importantes pero la cantidad de nulls es muy grande
        return df.drop(columns=['lat', 'lng', 'titulo', 'descripcion', 'id', 'idzona', 'direccion'], inplace=True)

    def drop_nan(self, df):
        return df.dropna(subset=['tipodepropiedad', 'provincia', 'ciudad'], inplace=True)

    def fill_metros(self, df):
        # Terreno null
        df1 = df[df.tipodepropiedad == 'Terreno'].fillna(0)
        df = df[df.tipodepropiedad != 'Terreno']
        df = pd.concat([df, df1])
        # Terreno != null
        cond1 = (df.tipodepropiedad == 'Terreno') & (df.metroscubiertos != 0)
        df.metroscubiertos = np.where(cond1, 0, df.metroscubiertos)

        # Apartamento
        cond = (df.tipodepropiedad == 'Apartamento') & (df.metrostotales.isnull())
        df.metrostotales = np.where(cond, df.metroscubiertos, df.metrostotales)

        # Metros totales = metros cubiertos SI es null
        df.metrostotales = np.where(df.metrostotales.isnull(), df.metroscubiertos, df.metrostotales)
        # Metros cubiertos = metros totales SI es null
        df.metroscubiertos = np.where(df.metroscubiertos.isnull(), df.metrostotales, df.metroscubiertos)

        df1 = df[df.tipodepropiedad == 'Terreno comercial'].fillna(0)
        df = df[df.tipodepropiedad != 'Terreno comercial']
        return pd.concat([df, df1])

    def predict_nulls(self, df):
        df = self.fill_xgboost(self, df, 'garages')
        df = self.fill_xgboost(self, df, 'habitaciones')
        df = self.fill_xgboost(self, df, 'banos')
        df = self.fill_xgboost(self, df, 'antiguedad', True)
        return df

    def fill_xgboost(self, df, feature, continua = False):
        # Columnas relevantes
        cols = ['tipodepropiedad', 'ciudad', 'provincia', 'antiguedad', 'habitaciones',
                'banos', 'metroscubiertos', 'metrostotales', 'fecha',
                'gimnasio', 'garages', 'usosmultiples', 'piscina', 'escuelascercanas',
                'centroscomercialescercanos', 'precio']
        # Usamos todas la que queremos predecir
        cols_subset = [x for x in cols if x != feature]

        df_train = df.dropna()
        df_test = df.loc[df[feature].isnull() == True]
        df_test = (df_test.dropna(subset=cols_subset))

        # Separamos los datos.
        df_train_x = df_train.loc[:, cols_subset]
        df_train_y = df_train[feature]
        df_test_y = df_test[feature]

        df_test_x = df_test.dropna(subset=cols_subset)

        df_test_x = df_test_x.loc[:, cols_subset]
        df_test = pd.merge(df_test_x, df_test_y, how='inner', left_index=True, right_index=True)
        df_test_x = df_test_x.loc[:, cols_subset]

        # Modelo - XGBoost
        params = {
            'min_child_weight': [5],
            'gamma': [1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'max_depth': [5],
            'n_estimators': [150, 250, 300, 500]}
        if continua:
            xgb = XGBRegressor(learning_rate=0.01,
                                silent=False, nthread=1)
        else:
            xgb = XGBClassifier(learning_rate=0.01,
                                silent=False, nthread=1)

        folds = 7
        param_comb = 4

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

        random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy',
                                           n_jobs=4,
                                           cv=skf.split(df_train_x, df_train_y), verbose=3, random_state=1001)

        # Here we go
        start_time = timer(None)  # timing starts from this point for "start_time" variable
        random_search.fit(df_train_x, df_train_y)
        self.timer(start_time)  # timing ends here for "start_time" variable

        print(random_search.score)

    def encoding(self, df, encoder):
        catlist = ['tipodepropiedad', 'ciudad', 'provincia']
        # 0 - Binario
        # 1 - One hot Encoding
        if encoder == 0:
            # Binary Encoding
            binary_enc = ce.BinaryEncoder()
            binary_encoded = binary_enc.fit_transform(df[catlist])
            df = df.join(binary_encoded.add_suffix('bc'))
        elif encoder == 1:
            # One hot Encoding
            one_hot_enc = ce.OneHotEncoder()
            one_hot_encoded = one_hot_enc.fit_transform(df[catlist])
            df = df.join(one_hot_encoded.add_suffix('oh'))

        return df.drop(columns=catlist, inplace=True)

    def features_engineering(self, df):
        # Partir fecha
        df.assign(
            day=df.fecha.dt.day,
            month=df.fecha.dt.month,
            year=df.fecha.dt.year)
        return df.drop(columns='fecha', inplace=True)
