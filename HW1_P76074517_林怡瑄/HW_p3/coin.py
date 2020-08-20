# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:36:48 2018

@author: Lisa
"""
import numpy as np
import matplotlib.pyplot as plt
import operator
import functools


PriorTheta = np.array([1/11 for i in range(11)])
#PriorTheta = [0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01]
thetarange = np.array([i for i in range(11)])/10
coin = ['h','h','t','t','t','t','t','t','t','t']

def c(n,k):
    if k==0:
        return 1
    else:
        return  functools.reduce(operator.mul, range(n - k + 1, n + 1)) /functools.reduce(operator.mul, range(1, k +1)) 

def Likelihood(Coin):
    Likelihood = []
    n=10
    k=0
    for i in Coin:
        if i=='h':
            k+=1
    for Theta in thetarange:
        p=c(n,k)
        for i in Coin:
            if i == 'h':
                p*=Theta
            else:
                p*=(1-Theta)
        Likelihood.append(p)
        
    
    return np.array(Likelihood)

def printbar(PriorTheta,Likelihood,Posterior):
    
    plt.title("PriorTheta")        
    plt.bar(thetarange, PriorTheta,width=0.05) 
    plt.show() 

    plt.title("Likelihood")    
    plt.bar(thetarange, Likelihood,width=0.05) 
    plt.show() 
  
    plt.title("Posterior")    
    plt.bar(thetarange, Posterior,width=0.05)  
    plt.show()  
    
def MAP_MLE(Likelihood,Posterior):
    i = Likelihood.tolist().index(np.max(Likelihood))
    MLE = thetarange[i]
    i = Posterior.tolist().index(np.max(Posterior))
    MAP = thetarange[i]
    return MAP,MLE
    

if __name__ == '__main__':
    
    c(10,0)
    
    PriorTheta = np.array([1/11 for i in range(11)])
    PriorX = np.sum(Likelihood(coin)*PriorTheta)
    Posterior = Likelihood(coin)*PriorTheta/PriorX    
    printbar(PriorTheta,Likelihood(coin),Posterior)
    MAP,MLE=MAP_MLE(Likelihood(coin),Posterior)
    print("MAP"+str(MAP))
    print("MLE"+str(MLE))
    
    PriorTheta = [0.01, 0.01, 0.05, 0.08, 0.15, 0.4, 0.15, 0.08, 0.05, 0.01, 0.01]
    PriorX = np.sum(Likelihood(coin)*PriorTheta)
    Posterior = Likelihood(coin)*PriorTheta/PriorX    
    printbar(PriorTheta,Likelihood(coin),Posterior)
    MAP,MLE=MAP_MLE(Likelihood(coin),Posterior)
    print("MAP"+str(MAP))
    print("MLE"+str(MLE))
    

    