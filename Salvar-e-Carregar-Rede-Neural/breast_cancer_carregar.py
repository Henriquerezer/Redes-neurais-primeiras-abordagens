# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:31:14 2022

@author: Henrique
"""

import numpy as np
import pandas as pd
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()
'''
carregando estrutura (modelo de classificação)
''' 
classificador = model_from_json(estrutura_rede)
'''
carregando os pesos (PODE DEMORAR UM POUCO)
'''
classificador.load_weights('classificador_breast.h5')

'''
Rede neural está pronta para ser utilizada, podemos testar em um novo registro
'''

'''
TESTAMOS EM UM RESGISTRO, EM QUE A REDE NEURAL, NÃO CONHECE ESSES DADOS.
'''
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsao = classificador.predict(novo)

'''
TAMBÉM PODEMOS TESTAR EM UM CONJUNTO DE DADOS 

Resalto que estou utilizando a mesma base de dados, utilizada ateriormente para o treinamento como é apenas para fins de aprendizado, não estou considerando isto.
mas para aplicações reais, o resultado deste teste não nos trará nenhum resultado relevante, pelo fato da rede neural ja conhecer esta base de dados 
'''

previsores = pd.read_csv('./entradas_breast.csv')
classe = pd.read_csv('./saidas_breast.csv')
classificador.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics = ['binary_accuracy'])
resultado  = classificador.evaluate(previsores,classe)

