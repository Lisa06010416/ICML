# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:35:07 2018

@author: Lisa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import *
import Layer
import MNIST_tools

MNIST_tools.downloadMNIST(path='MNIST_data', unzip=True)
x_train, y_train = MNIST_tools.loadMNIST(dataset="training", path="MNIST_data")
x_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path="MNIST_data")

def getFeature():
    
    layer_1 = Layer.Layer(784,128,"ReLu","h")
    layer_2 = Layer.Layer(128,784,"Sigmoid","o")
    
    #把weight拿出來
    layer_1.weight = np.load("p2_weight\\10_layer1weight.npy")
    layer_2.weight = np.load("p2_weight\\10_layer2weight.npy")
    
    # 計算
    layer_1.calculate(x_test)
    layer_2.calculate(layer_1.a)
    
    return layer_1.a

def Decoded_imgs():
    layer_1 = Layer.Layer(784, 128, "ReLu", "h")
    layer_2 = Layer.Layer(128, 784, "Sigmoid", "o")

    # 把weight拿出來
    layer_1.weight = np.load("p2_weight\\10_layer1weight.npy")
    layer_2.weight = np.load("p2_weight\\10_layer2weight.npy")

    # 計算
    layer_1.calculate(x_test)
    layer_2.calculate(layer_1.a)

    return layer_2.a

def Filter():
    layer_1 = Layer.Layer(784, 128, "ReLu", "h")
    layer_2 = Layer.Layer(128, 784, "Sigmoid", "o")

    # 把weight拿出來
    layer_1.weight = np.load("p2_weight\\10_layer1weight.npy")
    layer_2.weight = np.load("p2_weight\\10_layer2weight.npy")

    # 計算
    layer_1.calculate(x_test)
    layer_2.calculate(layer_1.a)

    return layer_1.weight

def plotScatter(X, title,minD):
    cm = plt.cm.get_cmap('RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=trainLabel, s=3,cmap=cm)
    plt.colorbar()
    plt.title(title)
    plt.show()

trainImages = getFeature()
trainLabel = y_test

# PCA
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(trainImages)
plotScatter(reduced_data_pca,"PCA",40)

# Visualize
decoded_imgs = Decoded_imgs()
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#filter
filter = np.delete(Filter().T,-1,axis=1)
n = 10
plt.figure(figsize=(10, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(filter[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()