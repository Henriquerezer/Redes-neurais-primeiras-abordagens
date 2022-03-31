# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:56:47 2022

@author: Henrique
"""

'''
Neste exemplo iremos classificar apenas um registro. O objeitvo é simular a ação de um médico inclundo dados de um exame de 
mamografia, e o algoritimo fará a classificação.

Os parâmetros da rede neural que escolheremos será baseado no resultaldo do grid_search, resultado do arquivo breast_cancer_tuning.
'''

import pandas as pd 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('./entradas_breast.csv')
classe = pd.read_csv('./saidas_breast.csv')

classificador = Sequential()
classificador.add(Dense(units = 8, activation= 'relu',
                        kernel_initializer= 'normal', input_dim= 30))
classificador.add(Dropout(0.2)) # Nesta linha foi implementado o DROPOUT, está ferramenta irá zerar uma taxa de neurônios de entradas ( Zerar seus valores), no caso foi escolhico que 0,2 (20%) dos neurônios deveriam ser zerados aleatóriamente na primeira camada.
classificador.add(Dense(units = 8, activation= 'relu',
                        kernel_initializer= 'normal'))
classificador.add(Dropout(0.2))# Nesta linha foi implementado o DROPOUT, está ferramenta irá zerar uma taxa de neurônios de entradas ( Zerar seus valores), no caso foi escolhico que 0,2 (20%) dos neurônios deveriam ser zerados aleatóriamente na segunda camada.
classificador.add(Dense(units = 1, activation = 'sigmoid')) #terceira camada


classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

'''
A variavel a baixo simula os dados de um novo paciente, os valores foram escolhidos aleatóriamente 
'''
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

'''
A etapa a seguir classifica os dados do novo paciente
'''
previsao = classificador.predict(novo)

'''
O valor abaixo de 0.5, é apenas um exemplo para zerar como resposta (TRUE OU FALSE), 
deve-se ter em mente que para o problema em questão deveriamos ter um critério maior
selecionando um valor mais próximo a (1)
'''
previsao1 = (previsao > 0.5)