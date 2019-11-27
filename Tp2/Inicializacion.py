# -*- coding: utf-8 -*-
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


class Inicializacion:
    # Encoder = 0 #Binario
    # Encoder = 1 #One Hot Encoding
    paramsGenerales = {
        'verbose': True,  # Agregar mas informacion. Texto, imagenes, guardar.
        'encoder': 0,  # Encoder. 0 Binario, 1 One hot encoding
        'esTest': False,  # Si es test no droppeamos nans
        # Para xgboost
        'scoring_regressor': 'neg_mean_squared_error',
        'scoring_cat': 'accuracy',
        'folds': 2,
        'param_comb': 1,
    }

    param_xgboost = {'max_depth': [8], 
        'colsample_bylevel': [0.6], 
        'colsample_bynode': [0.6], 
        'reg_alpha': [1.3000000000000003], 
        'n_estimators': [1300], 
        'colsample_bytree': [0.6], 
        'learning_rate': [0.09], 
        'gamma': [0.6000000000000001], 
        'min_child_weight': [3.5000000000000004], 
        'random_state': [42], 
        'verbosity': [0]}
    '''
    param_xgboost = {'n_estimators':np.arange(1200,1400,100),
        'learning_rate':[0.09,0.10,0.08,0.07],
        'gamma':np.arange(0,1,0.2),
        'colsample_bytree':[0.6],
        'colsample_bynode':[0.6],
        'colsample_bylevel':[0.6],
        'max_depth':np.arange(6,10,1),
        'min_child_weight':np.arange(0.5,5,0.3),
        'reg_alpha':np.arange(1,2,0.1),
        'random_state':[42],
        'verbosity':[0]
         }
    '''

    def __init__(self, params=[]):
        if params != []:
            self.paramsGenerales = params

        print("Inicializando")
        df = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')

        # Tiny dataset for debug purposes
        #df = df.sample(frac=0.001)
        #df_test = df_test.sample(frac=0.001)
        
        self.df_final = self.operaciones(df)

        self.paramsGenerales['esTest'] = True
        self.df_final_test = self.operaciones(df_test)

        # Salida en csv, para evitar recalculos
        self.df_final.to_csv('00-df_final.csv')
        self.df_final_test.to_csv('01-df_final_test.csv')

    def getDataframes(self):
        return self.df_final, self.df_final_test

    def operaciones(self, df):
        print("Comenzando operaciones")
        self.print_len(df)
        
        df = self.casteos(df)
        self.print_len(df)
        
        df = self.features_engineering(df)
        self.print_len(df)

        df = self.tratamiento_nulls(df)
        self.print_len(df)

        df = self.encoding(df)
        self.print_len(df)

        # Descartamos las columnas que fueron encodeadas
        print("   drop nans in selected columns")
        df = df.drop(columns=['tipodepropiedad', 'provincia', 'ciudad'])
        self.print_len(df)

        df = self.predict_nulls(df)
        self.print_len(df)

        self.features_engineering_v2(df)
        self.print_len(df)
        
        print('Qué columnas tienen nas?')
        print(df.isna().any())

        # Drop nan que quedaron pos limpieza y tratamientos
        if not self.paramsGenerales['esTest']:
            print("   drop other nans")
            df = df.dropna()
            self.print_len(df)

            df = self.recast(df)
            self.print_len(df)

            # self.metric_selection(df)
            # self.print_len(df)

        print('Columnas finales: ')
        print(df.columns)
        self.print_len(df)
        
        return df

    def print_len(self, df):
        if self.paramsGenerales['verbose']:
            print("Cantidad de registros: ", len(df))

    def casteos(self,df):
        cols = [x for x in df.columns if x!='precio' and df[x].dtype.type==np.float64]
        for x in cols:
            if df[x].isnull().sum() > 0:
                pass
            else:
                df[x] = df[x].astype(np.int16)
        df['fecha'] = df['fecha'].astype(np.datetime64)
        return df

    def tratamiento_nulls(self, df):
        print("   Nulls")
        df = self.drop_cols(df)
        df = self.fill_metros(df)
        return df

    def drop_cols(self, df):
        print("     Drop cols")
        return df.drop(columns=['id', 'lat', 'lng', 'titulo', 'descripcion', 'idzona', 'direccion'])
        # return df.drop(columns=['lat', 'lng', 'descripcion', 'idzona', 'direccion'])

    def drop_nan(self, df):
        print("     Drop nan")
        return df.dropna(subset=['tipodepropiedad', 'provincia', 'ciudad'])

    def fill_metros(self, df):
        print("     Fill metros")

        print("       Terreno")
        df1 = df[df.tipodepropiedad == 'Terreno'].fillna(0)
        df = df[df.tipodepropiedad != 'Terreno']
        df = pd.concat([df, df1])

        # Terreno != null
        cond1 = (df.tipodepropiedad == 'Terreno') & (df.metroscubiertos != 0)
        df.metroscubiertos = np.where(cond1, 0, df.metroscubiertos)

        print("       Apartamento")
        cond = (df.tipodepropiedad == 'Apartamento') & (df.metrostotales.isnull())
        df.metrostotales = np.where(cond, df.metroscubiertos, df.metrostotales)

        print("       Metros cubiertos null")
        # Metros totales = metros cubiertos SI es null
        df.metrostotales = np.where(df.metrostotales.isnull(), df.metroscubiertos, df.metrostotales)
        print("       Metros totales null")
        # Metros cubiertos = metros totales SI es null
        df.metroscubiertos = np.where(df.metroscubiertos.isnull(), df.metrostotales, df.metroscubiertos)

        print("       Terreno comercial")
        df1 = df[df.tipodepropiedad == 'Terreno comercial'].fillna(0)
        df = df[df.tipodepropiedad != 'Terreno comercial']

        df = pd.concat([df, df1])
        df = df.sample(frac=1)  # No entendí eso que tul
        return df

    def predict_nulls(self, df):
        print("     Predict nulls")

        f = [('garages', False), ('banos', False), ('habitaciones', False), ('antiguedad', True)]
        for feature, continua in f:
            df = self.fill_xgboost(df, feature, continua)
            if self.paramsGenerales['verbose']:
                self.mostrar_nulls(df, feature)
        return df

    def fill_xgboost(self, df, feature, continua=False):
        print("       fill with xgboost. Feature: ", {feature})

        # Columnas relevantes (Sin precio)
        cols = [x for x in df.columns if x != 'precio']
        # cols = [x for x in df.columns if x not in ['precio', 'titulo', 'tituloh']]

        # Todas - la que queremos predecir
        cols_subset = [x for x in cols if x != feature]

        start_time = self.timer(None)

        # TODO: Validar, si no existe el modelo, aunque quisiera usarlo no puede. Corregir eso.
        if self.paramsGenerales['esTest']:
            df_test_x = df.loc[:, cols_subset]

            random_search = pickle.load(open("models/00-nulls-xgb_" + feature + ".pickle", 'rb'))
            df_test_x[feature + '_xgb'] = random_search.predict(df_test_x)

        else:
            df_train = df.dropna()

            df_test = df.loc[df[feature].isnull() == True]
            # Separamos los datos.
            df_train_x = df_train.loc[:, cols_subset]
            df_train_y = df_train[feature]
            df_test_y = df_test[feature]

            df_test_x = df_test
            df_test_x = df_test_x.loc[:, cols_subset]
            df_test = pd.merge(df_test_x, df_test_y.to_frame(), how='inner', left_index=True, right_index=True)
            df_test_x = df_test_x.loc[:, cols_subset]

            # Hiperparametros - XGBoost

            folds = self.paramsGenerales['folds']
            param_comb = self.paramsGenerales['param_comb']

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
            if continua:
                xgb = XGBRegressor()
                scoring = self.paramsGenerales['scoring_regressor']
                cv = folds
            else:
                xgb = XGBClassifier()
                scoring = self.paramsGenerales['scoring_cat']
                cv = skf # 'accuracy'  #

            random_search = RandomizedSearchCV(xgb, param_distributions=Inicializacion.param_xgboost, n_iter=param_comb, scoring=scoring,
                                               n_jobs=-1,
                                               cv=cv, random_state=1001)

            random_search.fit(df_train_x, df_train_y)

            df_test_x[feature + '_xgb'] = random_search.predict(df_test_x)


        df = pd.merge(df, df_test_x[feature + '_xgb'].to_frame(), how='left', left_index=True, right_index=True)
        df[feature] = np.where((df[feature].isnull()), df[feature + '_xgb'], df[feature])
        df.drop(columns=[feature + '_xgb'], inplace=True)

        self.timer(start_time)

        if not self.paramsGenerales['esTest']:
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
            print('         Los mejores parametros son: ')
            print('        ', best_pars)
            print('         ------------------------')
            print('         Best score: ')
            print('        ', random_search.best_score_)
            print('         ------------------------')

            # Best XGB model that was found based on the metric score you specify
            best_model = random_search.best_estimator_
            # Save model
            pickle.dump(random_search.best_estimator_, open("models/00-nulls-xgb_" + feature + ".pickle", "wb"))

        return df

    def encoding(self, df):
        print("   Encoding")
        catlist = ['tipodepropiedad', 'ciudad', 'provincia']

        # 0 - Binario
        # 1 - One hot Encoding
        if self.paramsGenerales['encoder'] == 0:
            if self.paramsGenerales['esTest']:
                binary_encoded = self.binary_enc.transform(df[catlist])
                df = df.join(binary_encoded.add_suffix('bc'))
            else:
                self.binary_enc = ce.BinaryEncoder()
                binary_encoded = self.binary_enc.fit_transform(df[catlist])
                df = df.join(binary_encoded.add_suffix('bc'))


        elif self.paramsGenerales['encoder'] == 1:
            if self.paramsGenerales['esTest']:
                one_hot_encoded = self.one_hot_enc.transform(df[catlist])
                df = df.join(one_hot_encoded.add_suffix('oh'))
            else:
                self.one_hot_enc = ce.OneHotEncoder()
                one_hot_encoded = self.one_hot_enc.fit_transform(df[catlist])
                df = df.join(one_hot_encoded.add_suffix('oh'))

        if not self.paramsGenerales['esTest']:
            self.tiposPrincipales = df.tipodepropiedad.value_counts().sort_values(kind="quicksort", ascending=False).to_frame()
            # print(self.tiposPrincipales.head(5).index.tolist())
            self.tiposPrincipales = self.tiposPrincipales.head(5).index.tolist()
            # print(self.tiposPrincipales)
        print('   Encoding top 5 propiedades')
        for tipo in self.tiposPrincipales:
            print('     ' + tipo.replace(' ', '_'))
            df['es_' + tipo.replace(' ', '_')] = df.tipodepropiedad == tipo

        # TODO: Mix encoding
        # TODO: Encoding con promedio

        # TODO: Otros encodings?
        # TODO: AUC Scoring?
        return df

    def features_engineering(self, df):
        print("   Features engineering")

        print("     Fecha")
        df['anio'] = df.fecha.dt.year

        # http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
        df['mes_sin'] = np.sin((df.fecha.dt.month - 1) * (2. * np.pi / 12))
        df['mes_cos'] = np.cos((df.fecha.dt.month - 1) * (2. * np.pi / 12))

        print("     Amennities")
        df['amenities'] = df.piscina + df.gimnasio + df.usosmultiples
        print("     Ubicacion")
        df['ubicacion'] = df.escuelascercanas + df.centroscomercialescercanos


        # features basadas en str.contains sobre la descripción
        zonas_exclusivas = ['Lomas de Chapultepec', 'Polanco', 'Bosques de las Lomas', 'Colinas del Bosque', 'Jardines del Pedregal',
                   'Lomas Virreyes', 'Tlalpuente', 'Jardines en la Montaña', 'Bosques de Tlalpan']
        refaccion = ['a reparar', 'a refaccionar','a restaurar']
        lujo = ['marmol', 'mansión', 'lujo', 'jacuzzi']
        vigilancia = ['vigilancia'] 
        country = ['country', 'barrio privado']
        
        print("     Features basadas en la descripción")
        df['zonas_exclusivas'] = df.descripcion.str.contains('|'.join(zonas_exclusivas), na=False, case=False)
        df['refaccion'] = df.descripcion.str.contains('|'.join(refaccion), na=False, case=False)
        df['lujo'] = df.descripcion.str.contains('|'.join(lujo), na=False, case=False)
        df['vigilancia'] = df.descripcion.str.contains('|'.join(vigilancia), na=False, case=False)
        df['country'] = df.descripcion.str.contains('|'.join(country), na=False, case=False)

        # df = df.assign(
        #     day=df.fecha.dt.day,
        #     month=df.fecha.dt.month,
        #     year=df.fecha.dt.year)

        df = df.drop(columns='fecha')

        return df

    def features_engineering_v2(self, df):
        print("     patio")
        df['patio'] = df.metrostotales - df.metroscubiertos
        print("     cantidad de ambientes(incluye baño)")
        df['ambientes'] = df.habitaciones + df.banos + df.garages
        print('nulls en divisor:')

        # SACAR CUANDO FIXEEMOS EL XGBOOST que predice ceros...
        df.ambientes.replace(0.0, df.ambientes.median(), inplace=True)
        df.metroscubiertos.replace(0.0, df.metroscubiertos.median(), inplace=True)
        df.metrostotales.replace(0.0, df.metrostotales.median(), inplace=True)

        print("     tamaño promedio del ambiente")
        df['prom_amb'] = df.metroscubiertos / df.ambientes
        print('     densidad de construccion')
        df['construccion_density'] = df.metroscubiertos/df.metrostotales

        print(df.prom_amb.isnull().values.any())
        print(df.construccion_density.isnull().values.any())
        
        return df 



    def metric_selection(self, df):
        print("   Metric selection")
        feature_cols = df.columns.drop('precio')
        train, valid, _ = self.get_data_splits(df)

        print('     f_classif')
        start_time = self.timer(None)
        selector = SelectKBest(f_classif, k=3)

        X_new = selector.fit_transform(train[feature_cols], train['precio'])
        self.print_metrics(feature_cols, selector.scores_, 'f_classif')
        self.timer(start_time)

        # print('     chi2')
        # start_time = self.timer(None)

        # selector_chi2 = SelectKBest(chi2, k=3)
        # X_new = selector_chi2.fit_transform(train[feature_cols], train['precio'])
        # self.print_metrics(feature_cols, selector_chi2.scores_, 'chi2')
        #
        # # Sin la primera
        # plt.bar(feature_cols[1:], selector_chi2.scores_[1:])
        # self.print_metrics(feature_cols[1:], selector_chi2.scores_[1:], 'chi2 - Sin primera feature')
        # self.timer(start_time)

        print('     mutual_info_classif')
        start_time = self.timer(None)
        selector_mutual = SelectKBest(mutual_info_classif, k=3)

        X_new = selector_mutual.fit_transform(train[feature_cols], train['precio'])
        self.print_metrics(feature_cols, selector_mutual.scores_, 'mutual_info_classif')
        self.timer(start_time)

        print('     L1 regularization (Lasso Regression)')
        start_time = self.timer(None)
        train, valid, _ = self.get_data_splits(df)
        X, y = train[train.columns.drop("precio")], train['precio']

        # Set the regularization parameter C=1
        logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(X, y)
        self.print_metrics(feature_cols, logistic.coef_[0], 'regulatization c=1')
        self.timer(start_time)

        print('     Random forest')
        start_time = self.timer(None)
        X, y = train[train.columns.drop("precio")], train['precio']
        val_X, val_y = valid[valid.columns.drop("precio")], valid['precio']

        forest_model = RandomForestClassifier(random_state=1)
        forest_model.fit(X, y)
        preds = forest_model.predict(val_X)

        self.print_metrics(feature_cols, forest_model.feature_importances_, 'Random forest')
        self.timer(start_time)

    def print_metrics(self, feature_cols, selector, modelo):
        plt.bar(feature_cols, selector)
        plt.xlabel('Features')
        plt.xticks(rotation=90)
        plt.ylabel('Importancia')
        plt.title('Importancia Features con Univariate (' + modelo + ')')
        plt.show()

    def timer(self, start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('       Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

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

            plt.savefig(plots_dir + file_name + '.png')

    def get_data_splits(self, df, valid_fraction=0.1):
        valid_size = int(len(df) * valid_fraction)

        train = df[:-valid_size * 2]
        valid = df[-valid_size * 2:-valid_size]
        test = df[-valid_size:]

        return train, valid, test

    def recast(self, df):
        print("   Recast final")
        cols = [x for x in df.columns if x!='precio' and x!='mes_sin' and x!='mes_cos' and x!='construccion_density' and x!='prom_amb' and df[x].dtype.type == np.float64]
        print(cols)
        for x in cols:
            if df[x].isnull().sum() > 0:
                pass
            else:
                df[x] = df[x].astype(np.int16)
        return df
        #try:
        #    columns.remove('precio')
        #    columns.remove('mes_sin')
        #    columns.remove('mes_cos')
        #except:
            #pass

        #for x in range(len(columns)):
           # dtype = df[columns[x]].dtype.type
          #  if dtype == np.int64 or dtype == np.float64:
         #       df = df.astype({columns[x]: np.int16})
        #return df


if __name__ == '__main__':
    preprocesamiento = Inicializacion()
