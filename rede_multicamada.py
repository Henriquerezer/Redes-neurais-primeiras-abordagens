# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:07:22 2022

@author: Henrique
"""
##FEET-FOWARD
import numpy as np

def sigmoid(soma):
    return 1 /(1+np.exp(-soma)) #exp - exponencial

def sigmoidDerivada(sig):
    return sig * (1 - sig)

#a = sigmoid(0.5) testes
#b = sigmoidDerivada(a) testes

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
saidas = np.array([[0],[1],[1],[0]])

#pesos0 = np.array([[-0.424, -0.740, -0.961],
                 #  [0.358, - 0.577, -0.469]])

#pesos1 = np.array([[-0.017], [-0.893],[0.148]])

pesos0 = 2*np.random.random((2,3)) - 1 #iniciamos deste modo, para que os pesos
pesos1 = 2*np.random.random((3,1)) - 1 # sejam iniciados de forma aleatória, não de maneira fixa como na linha de cima
epocas = 1000000
taxadeAprendizagem = 0.6
momento = 1

for j in range (epocas):
    camamaentrada = entradas
    somaSinapse0 = np.dot(camamaentrada, pesos0)
    camadaoculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaoculta, pesos1)
    camadadesaida = sigmoid(somaSinapse1)
    
    errocamadaSaida = saidas - camadadesaida
    mediaabsoluta = np.mean(np.abs(errocamadaSaida))
    print("erro: " + str(mediaabsoluta))
    
    derivadasaida = sigmoidDerivada(camadadesaida)
    deltasaida = errocamadaSaida * derivadasaida
    
    pesos1Transposta = pesos1.T
    deltasaidaXpeso = deltasaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltasaidaXpeso * sigmoidDerivada(camadaoculta)
    
    camadaocultaTransposta = camadaoculta.T
    pesosNovo1 = camadaocultaTransposta.dot(deltasaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxadeAprendizagem)
    
    camadaentradaTransposta = camamaentrada.T
    pesosNovo0 = camadaentradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxadeAprendizagem)
    
    
    