# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:46:55 2022

@author: Henrique
"""

from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import numpy as np
from keras.preprocessing import image


'''
1° camada de convolução
'''
classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
'''
2° camada de convolução
'''
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
'''
Transforma a matriz em VETOR
'''
classificador.add(Flatten())

'''
ESTRUTURA DA REDE NEURAL DENSA
'''
# 1° CAMADA OCULTA
classificador.add(Dense(units = 128, activation = 'relu',))
classificador.add(Dropout(0.2))
#2° CAMADA OCULTA
classificador.add(Dense(units = 128, activation = 'relu',))
classificador.add(Dropout(0.2))
#CAMADA DE SAÍDA
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64,64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')

base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'binary')


classificador.fit_generator(base_treinamento, steps_per_epoch = 4000/32,
                            epochs = 15, validation_data = base_teste,
                            validation_steps= 1000/32)

imagem_teste = image.load_img('dataset/test_set/gato/cat.3500.jpg',
                              target_size = (64,64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)

previsao = (previsao > 0.5)

if previsao > 0.5:
    print('É UM GATO')
else:
    print('É UM CACHORRO')
    
    
#base_treinamento.class_indices
# Classe 0 -> GATO
#Classe 1  -> CACHORRO







