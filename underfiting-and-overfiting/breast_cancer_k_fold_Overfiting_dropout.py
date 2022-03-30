# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:13:49 2022

@author: Henrique
"""

"""
Neste exercício será utilizada a técnica de DROPOUT com o objetivo de diminuir o overfiting do nosso modelo de classificação
este exercício é o mesmo publicado anteriormente, porém sem a diminuição do overfiting.

É recomendado que por padrão já se utilize essa ferramenta, pois diminuira o overfiting do modelo, devemos lembrar que a rede neural 
possui diferças opções de ajustes, então a chance de overfiting é sempre alta.
"""
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv('./entradas_breast.csv')
classe = pd.read_csv('./saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation= 'relu',
                            kernel_initializer= 'random_uniform', input_dim= 30))
    classificador.add(Dropout(0.2)) # Nesta linha foi implementado o DROPOUT, está ferramenta irá zerar uma taxa de neurônios de entradas ( Zerar seus valores), no caso foi escolhico que 0,2 (20%) dos neurônios deveriam ser zerados aleatóriamente na primeira camada.
    classificador.add(Dense(units = 16, activation= 'relu',
                            kernel_initializer= 'random_uniform'))
    classificador.add(Dropout(0.2))# Nesta linha foi implementado o DROPOUT, está ferramenta irá zerar uma taxa de neurônios de entradas ( Zerar seus valores), no caso foi escolhico que 0,2 (20%) dos neurônios deveriam ser zerados aleatóriamente na segunda camada.
    classificador.add(Dense(units = 1, activation = 'sigmoid',)) #terceira camada

    opt = keras.optimizers.Adam(learning_rate=0.001, decay = 0.0001, clipvalue = 0.5)

    classificador.compile(opt, loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador 

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 300,
                                batch_size = 5)
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvioPadrao = resultados.std()