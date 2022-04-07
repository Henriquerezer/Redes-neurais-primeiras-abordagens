# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:09:29 2022

@author: Henrique
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:29:40 2022

@author: Henrique
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
'''
Foi realizada a pesquisa por uma nova biblioteca para a importação desta função.
A operação fornecida ->  from keras.layers.normalization import BatchNormalization
Não funciona mais, indico utilizar o método abaixo.
'''
from tensorflow.keras.layers import BatchNormalization

(X_treinamento, y_treinamento) , (X_teste , y_teste) = mnist.load_data()
plt.imshow(X_treinamento[0], cmap = 'gray') #somente visualização
plt.title('Classe' + str(y_treinamento[0])) 

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 
                                   28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#normalização dos valores, unida a float32
previsores_treinamento /= 255 #1 byte de 0 a 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste =  np_utils.to_categorical(y_teste, 10)

# Estrututa rede neural

#1° OPERADOR DE CONVULUÇÃO
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (28,28,1),
                         activation= 'relu')) #pratica recomendada começar com 64
classificador.add(BatchNormalization())

#2° POOLING

classificador.add(MaxPooling2D(pool_size = (2,2)))


#3° FLATTENING

#classificador.add(Flatten()) ESTÁ FLATTEN VAI FICAR COMENTADO PO NÃO QUEREMOS NA PRÓXIMA LINHA 
#DO CÓDIGO AINDA PRECISAREMOS TRABALHAR COM UMA MATRIZ.

# Mais uma camada de convolução

classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())

#Segunda camada de POOLING
classificador.add(MaxPooling2D(pool_size = (2,2)))
'''
# Segunda Camada de FLATTENING NÃO POSSO UTILIZAR MAIS DE UMA VEZ, TANTO EM PROBLEMAS COM UMA CAMDA OCULTA 
#OU MAIS DE UMA CAMADA OCULTA. PELO FATO DO FLATTEN TRANSFORAR A MATRIZ EM UM VETOR, E SÓ PODEMOS TRABALHAR COM VETOR 
# UMA ÚNICA VEZ LOGO AO ENTRAR NA ESTRUTURA DA REDE NEURAL
'''
classificador.add(Flatten())

#4° GERAR REDE NEURAL DENSA

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 5,
                  validation_data = (previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)





