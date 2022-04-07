# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:36:24 2022

@author: Henrique
"""

from keras.datasets import mnist 
from keras.models import Sequential 
import numpy as np
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import StratifiedKFold

seed = 6
np.random.seed(seed)

(X, y), (X_teste, y_test) = mnist.load_data()
previsores = X.reshape(X.shape[0], 28,28, 1)
previsores = previsores.astype('float32')
previsores /= 255
classe = np_utils.to_categorical(y,10)

kfold = StratifiedKFold(n_splits = 6, shuffle= True, random_state = seed)
resultados = [] # lista vazia para colocarmos os resultados de cada execução

a =  np.zeros(5)
b = np.zeros(shape = (classe.shape[0],1))

for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape = (classe.shape[0],1))):
    #print('Indices treinamento:', indice_treinamento, 'Indice teste', indice_teste)
    #1° OPERADOR DE CONVULUÇÃO
    classificador = Sequential()
    classificador.add(Conv2D(32, (3,3), input_shape = (28,28,1),
                             activation= 'relu')) #pratica recomendada começar com 64

    #2° POOLING

    classificador.add(MaxPooling2D(pool_size = (2,2)))

    #3° FLATTENING

    classificador.add(Flatten())

    #4° GERAR REDE NEURAL DENSA

    classificador.add(Dense(units = 128, activation = 'relu'))
    classificador.add(Dense(units = 10, activation = 'softmax'))
    classificador.compile(loss = 'categorical_crossentropy',
                          optimizer = 'adam', metrics = ['accuracy'])
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento],
                      batch_size = 128, epochs = 5)
    precisao = classificador.evaluate(previsores[indice_teste], classe[indice_teste])
    resultados.append(precisao[1])

media = sum(resultados) / len(resultados)








