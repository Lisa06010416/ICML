# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:19:37 2018

@author: Lisa
"""


import numpy as np
import coin
import random
import math
import matplotlib.pyplot as plt

PriorTheta = np.array([1/11 for i in range(11)])

def simulator():  
    tencoin = []
    for i in range(10):
        if  random.randrange(0,2) == 0:
            tencoin.append('h')
        else :
            tencoin.append('t')
    return tencoin


tencoin = simulator()
entropy=[]

for i in range(1,51):
    tencoin = simulator()
    Likelihood = coin.Likelihood(tencoin)
    PriorX = np.sum(Likelihood*PriorTheta)
    Posterior = Likelihood*PriorTheta/PriorX
    PriorTheta = Posterior
    e=0
    for j in Posterior:
        if(j!=0):
            e=e-j*math.log(j,2)
    entropy.append(e)    
    if i%10 == 0:
        print("After "+str(i))
        coin.printbar(PriorTheta,Likelihood,Posterior)
        MAP,MLE=coin.MAP_MLE(Likelihood,Posterior)
        print("MAP"+str(MAP))
        print("MLE"+str(MLE))
        
plt.plot(entropy)
    

        