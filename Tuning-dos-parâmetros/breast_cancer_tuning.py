# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:21:15 2022

@author: Henrique
"""

import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


previsores = pd.read_csv('./entradas_breast.csv')
classe = pd.read_csv('./saidas_breast.csv')

def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation= activation,
                            kernel_initializer= kernel_initializer, input_dim= 30))
    classificador.add(Dropout(0.2)) # Nesta linha foi implementado o DROPOUT, está ferramenta irá zerar uma taxa de neurônios de entradas ( Zerar seus valores), no caso foi escolhico que 0,2 (20%) dos neurônios deveriam ser zerados aleatóriamente na primeira camada.
    classificador.add(Dense(units = neurons, activation= activation,
                            kernel_initializer= kernel_initializer))
    classificador.add(Dropout(0.2))# Nesta linha foi implementado o DROPOUT, está ferramenta irá zerar uma taxa de neurônios de entradas ( Zerar seus valores), no caso foi escolhico que 0,2 (20%) dos neurônios deveriam ser zerados aleatóriamente na segunda camada.
    classificador.add(Dense(units = 1, activation = 'sigmoid')) #terceira camada


    classificador.compile(optimizer = optimizer, loss = loss,
                          metrics = ['binary_accuracy'])
    return classificador 

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [1,10],
              'epochs': [50,200],
              'optimizer': ['adam','sgd'],
              'loss': ['binary_crossentropy','hinge'],
              'kernel_initializer': ['random_uniform','normal'],
              'activation': ['relu','tanh'],
              'neurons':[16, 8]}

grid_search = GridSearchCV(estimator = classificador, 
                           param_grid = parametros,
                           scoring= 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_


'''
TESTAR MAIS TARDE E COLOCAR NO GITHUB
'''




