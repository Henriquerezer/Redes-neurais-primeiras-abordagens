# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:26:36 2022

@author: Henrique
"""
import numpy as np
entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

def soma(e, p):
   return e.dot(p)
#dot product = Produto escalar
    
s = soma(entradas, pesos)
def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0
r = stepFunction(s)