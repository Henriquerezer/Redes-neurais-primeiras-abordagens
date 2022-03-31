# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:16:22 2022

@author: Henrique
"""
import pandas as pd 

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('./entradas_breast.csv')
classe = pd.read_csv('./saidas_breast.csv')

classificador = Sequential()
classificador.add(Dense(units = 8, activation= 'relu',
                        kernel_initializer= 'random_uniform', input_dim= 30))
classificador.add(Dropout(0.2)) # Nesta linha foi implementado o DROPOUT, está ferramenta irá zerar uma taxa de neurônios de entradas ( Zerar seus valores), no caso foi escolhico que 0,2 (20%) dos neurônios deveriam ser zerados aleatóriamente na primeira camada.
classificador.add(Dense(units = 8, activation= 'relu',
                        kernel_initializer= 'random_uniform'))
classificador.add(Dropout(0.2))# Nesta linha foi implementado o DROPOUT, está ferramenta irá zerar uma taxa de neurônios de entradas ( Zerar seus valores), no caso foi escolhico que 0,2 (20%) dos neurônios deveriam ser zerados aleatóriamente na segunda camada.
classificador.add(Dense(units = 1, activation = 'sigmoid')) #terceira camada


classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size = 10, epochs = 200)

'''
VAMOS SALVAR EM DISCO
'''
classificador_json = classificador.to_json() # Estrutura da rede neural
with open('classificador_breast.json','w') as json_file:
    json_file.write(classificador_json) #salvando a estrutura em disco
'''
Vamos salvar os pesos no formato h5, não esqueça de instalar o pacote h5py
'''
classificador.save_weights('classificador_breast.h5')    



