import datetime
import itertools
import os
from random import randint

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, BayesianRidge, HuberRegressor
from lightgbm import Dataset
from lightgbm import train
from lightgbm import cv

from mlxtend.classifier import StackingClassifier
from mlxtend.plotting import plot_learning_curves
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor


class Ensambles:
    paramsGenerales: {
        'verbose': True
    }

    def __init__(self, modelos=[], dataframes=[], params=[]):
        if params != []:
            self.paramsGenerales = params

        # https://github.com/vsmolyakov/experiments_with_python/blob/master/chp01/ensemble_methods.ipynb
        df_train = pd.read_csv('00-df_final.csv')
        # df_train = df_train.sample(frac=0.1)
        # self.mostrar_nulls(df_train)

        X, y = df_train[df_train.columns.drop("precio")], df_train.precio
        # print(X)
        # construccion_density
        # print("Decimales mayores al max de float64")
        # print(np.where(X.values >= np.finfo(np.float64).max))

        bayesrCV = pickle.load(open("models/bayesrCV.pkl", 'rb'))
        catboost = pickle.load(open("models/catboost.pkl", 'rb'))
        enetCV = pickle.load(open("models/enetCV.pkl", 'rb'))
        hrCV = pickle.load(open("models/hrCV.pkl", 'rb'))
        LassoCV = pickle.load(open("models/LassoCV.pkl", 'rb'))
        lightgbm = pickle.load(open("models/lightgbm.pkl", 'rb'))
        RidgeCV = pickle.load(open("models/RidgeCV.pkl", 'rb'))
        svrCV = pickle.load(open("models/svrCV.pkl", 'rb'))
        xgboost = pickle.load(open("models/xgboost.pkl", 'rb'))

        lr = XGBRegressor()
        clf_list = [bayesrCV, catboost, enetCV, hrCV, LassoCV, lightgbm, RidgeCV, svrCV, xgboost]
        sclf = StackingClassifier(classifiers=clf_list,
                                  meta_classifier=lr)
        clf_list.append(sclf)

        label = [
            'bayesrCV.pkl'
            , 'catboost.pkl'
            , 'enetCV.pkl'
            , 'hrCV.pkl'
            , 'LassoCV.pkl'
            , 'lightgbm.pkl'
            , 'RidgeCV.pkl'
            , 'svrCV.pkl'
            , 'xgboost.pkl'
            , 'Stacking'
        ]

        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2)
        grid = itertools.product([0, 1], repeat=2)

        start_time = self.timer(None)

        clf_cv_mean = []
        clf_cv_std = []
        for clf, label, grd in zip(clf_list, label, grid):
            scores = cross_val_score(clf, X, y, cv=3, scoring='neg_mean_absolute_error')
            print("Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))
            clf_cv_mean.append(scores.mean())
            clf_cv_std.append(scores.std())

            clf.fit(X, y)

        self.timer(start_time)

        # ax = plt.subplot(gs[grd[0], grd[1]])
        # fig = plot_decision_regions(X=X, y=y, clf=clf)
        # plt.title(label)

        # plt.show()

        self.PlotAccuracy(clf_cv_mean, clf_cv_std, label)
        self.PlotLearningCurve(X, sclf, y)

        self.TransformTest(clf)
        # We can see that stacking achieves higher accuracy than individual classifiers and based on learning curves, it shows no signs of overfitting.

    def TransformTest(self, clf):
        test = pd.read_csv('00-df_final_test.csv')

        # self.predicted = clf.transform(test)
        self.predicted = clf.predict(test)

        self.test_ids = pd.read_csv('01-df_final_test.csv')['id']
        answer = pd.DataFrame(list(zip(self.test_ids, self.predicted)), columns=['id', 'target'])
        answer.to_csv('{}-{}.csv'.format('z-result-ensamble', int(round(self.score))), sep=',', index=False)



    def PlotLearningCurve(self, X, sclf, y):
        # plot learning curves
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        plt.figure()
        plot_learning_curves(X_train, y_train, X_test, y_test, sclf, print_model=False, style='ggplot')
        plt.show()

    def PlotAccuracy(self, clf_cv_mean, clf_cv_std, label):
        # plot classifier accuracy
        plt.figure()
        (_, caps, _) = plt.errorbar(range(4), clf_cv_mean, yerr=clf_cv_std, c='blue', fmt='-o', capsize=5)
        for cap in caps:
            cap.set_markeredgewidth(1)
        plt.xticks(range(4), label)
        plt.ylabel('Accuracy')
        plt.xlabel('Classifier')
        plt.title('Stacking Ensemble')
        plt.show()

    def mostrar_nulls(self, df, feature=randint(1, 101)):
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
            file_name = str(feature)

            if not os.path.isdir(plots_dir):
                os.makedirs(plots_dir)
            plt.show()
            plt.savefig(plots_dir + file_name + '.png')

    def timer(self, start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('       Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
