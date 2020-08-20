# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 18:43:35 2018

@author: Lisa
"""

import urllib.request
import os
import re
from keras.preprocessing.text import Tokenizer
import numpy as np
import csv
import math

def  pre2(topK):
   
    trainX = []
    testX = []
    trainY = np.load("y_train.npy")
    testY = np.load("y_test.npy")
    
    for i in np.load("x_train.npy"):
        trainX.append([ x for x in i if x <topK+1])
    for i in np.load("x_test.npy"):
        testX.append([ x for x in i if x<topK+1])
      
    return trainX,testX,trainY,testY


def writefile(path,Table):
    with open(path,'w') as f :
        writer = csv.writer(f)
        
        row = [i for i in range(1,101)]
        row.insert(0,'topK')
        writer.writerow(row)
        
        row = list(Table[0])
        row[0] = 'N'
        writer.writerow(row)
        
        row = list(Table[1])
        row[0] = 'P'
        writer.writerow(row)
        
def tftable(P,N,y,testY):
    TF_table=[0,0,0,0]  # TP , FP , FN , TN
    for i in range(len(y)):
        if y[i] == testY[i] :
            if y[i] == P:   #TP
                TF_table[0]+=1
            else : #TN
                TF_table[3]+=1
        else:
            if y[i] == P:   #FP
                TF_table[1]+=1
            else: #FN
                TF_table[2]+=1
    return TF_table

  
topK = 10000
trainX,testX,trainY,testY = pre2(topK)
 
#count Likelihood Table 
CountTable=[[0 for i in range(topK+1)],[0 for i in range(topK+1)]]

print("count Likelihood Table ...")
for i in range(1,len(CountTable[0])):
    for j in range(len(trainX)):       
        if (i in trainX[j]):
            CountTable[trainY[j]][i]+=1
ps=0
ns=0            
for i in trainY:
    if i==1 :
        ps+=1
    else :
        ns+=1

CountTable = np.array(CountTable)
#s = CountTable.sum(axis=1)
#print(s)
LikelihoodTable = [CountTable[0]/ns,CountTable[1]/ps] 
writefile('LikelihoodTable'+str(topK)+'.csv',LikelihoodTable)

LikelihoodTable = np.array(LikelihoodTable)
#sum1 = [s[0]/(s[0]+s[1]),s[1]/(s[0]+s[1])]
#sum2 = CountTable.sum(axis=0)/(s[0]+s[1])
#print(sum1)

#predict on testX
y=[]
for i in testX:
    p=0
    for j in range(1,len(LikelihoodTable[0])) :
        if(j in i):
            p=p+math.log(LikelihoodTable[1][j])
    p=p+math.log(ps/(ns+ps))
    
    n=0
    for j in range(1,len(LikelihoodTable[0])) :
        if(j in i):
            n=n+math.log(LikelihoodTable[0][j])
    n=n+math.log(ns/(ns+ps))
    
    if n>p:
        y.append(0)
    else:
        y.append(1)

with open("predict"+str(topK)+".csv",'w') as f:
    writer = csv.writer(f)
    for i in range(len(y)):
        writer.writerow([y[i],testY[i]])
        
#accuracy, precision, and  recall 
        
T=0
F=0
for i in range(len(y)):
    if y[i] == testY[i] :
        T+=1
    else:
        F+=1
        
print("accuracy : "+str(T/(F+T)))
    
TF_table = tftable(1,0,y,testY)
print("             precision            recall ")
print("positive : "+str(TF_table[0]/(TF_table[0]+TF_table[1]))+"    "+str(TF_table[0]/(TF_table[0]+TF_table[2])))

TF_table = tftable(0,1,y,testY)
print("negative : "+str(TF_table[0]/(TF_table[0]+TF_table[1]))+"    "+str(TF_table[0]/(TF_table[0]+TF_table[2])))
