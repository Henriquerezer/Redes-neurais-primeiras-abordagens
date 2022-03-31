# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:19:07 2022

@author: Henrique
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:,4].values
'''
Transformação dos dados de classe de string para valores numéricos 
'''
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
#IRIS SETOSA 1 0 0
#IRIS VIRGINICA 0 1 0 
#IRIS VERSICOLOR 0 0 1

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation= 'relu', input_dim = 4))
    classificador.add(Dense(units = 4, activation = 'relu'))
    classificador.add(Dense(units = 3, activation = 'softmax')) #estamos utilizando está função pq o problema de classificação nos resulta em mais de 2 classes 
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn= criar_rede,
                                epochs = 1000,
                                batch_size = 10)

resultados = cross_val_score(estimator = classificador, X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()


'''
Atividade -> ALTERAR OS PARÂMETROS E ANALISAR OS RESULTADOS
             FAZER O TUNING COM GRIDSEARCH E TESTAR COM OS RESULTADOS 
'''



