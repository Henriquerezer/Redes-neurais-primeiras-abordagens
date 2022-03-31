# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:01:44 2022

@author: Henrique
"""
## Data base -> https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29  ##
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv('./entradas_breast.csv')
classe = pd.read_csv('./saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation= 'relu',
                            kernel_initializer= 'random_uniform', input_dim= 30))
    classificador.add(Dense(units = 16, activation= 'relu',
                            kernel_initializer= 'random_uniform'))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=0.001, decay = 0.0001, clipvalue = 0.5)

    classificador.compile(optimizer = opt, loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador 

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 1000,
                                batch_size = 10)
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvioPadrao = resultados.std()
