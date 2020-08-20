# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:37:52 2018

@author: Lisa
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.decomposition import *
from sklearn.manifold import LocallyLinearEmbedding
from matplotlib import offsetbox

def decode_idx3_ubyte(idx3_ubyte_file):

    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    _, nums, height, width = struct.unpack_from(fmt_header, bin_data, offset)
    print ("圖數量 :"+str(nums))
    print("圖形大小 :"+str(height)+"x"+str(width))

    image_size = height * width
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((nums, image_size))
    for i in range(nums):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):

    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    _, nums = struct.unpack_from(fmt_header, bin_data, offset)
    print ("label數量 :"+str(nums))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(nums)
    for i in range(nums):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def plotScatter(X, title,minD):
    
    
    cm = plt.cm.get_cmap('RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=trainLabel, s=3,cmap=cm)
    plt.colorbar()
    plt.title(title)
    plt.show()
    

    plt.figure()
    ax = plt.subplot(111)
    cm = plt.cm.get_cmap('RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=trainLabel, s=3,cmap=cm)
    plt.colorbar()
    
    shown_images = np.array([[1., 1.]])  
    for i in range(trainImages.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < minD:
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        
        if(trainLabel[i]==2):
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(trainImages[i].reshape(28, 28)),X[i])
            ax.add_artist(imagebox)
    
    plt.title(title)
    plt.show()

trainImages = decode_idx3_ubyte("train-images.idx3-ubyte")
trainLabel = decode_idx1_ubyte("train-labels.idx1-ubyte")

for i in range(10):
    ax = plt.subplot(1, 10, i+1)
    plt.imshow(trainImages[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(str(trainLabel[i]))
plt.show()


#PCA
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(trainImages)

plotScatter(reduced_data_pca,"PCA",40000)

#ICA
ica = FastICA(n_components=2)
reduced_data_ica = ica.fit_transform(trainImages)

plotScatter(reduced_data_ica,"ICA",0.000001)

#LLE
lle = LocallyLinearEmbedding(n_components=2)
reduced_data_lle = lle.fit_transform(trainImages)

plotScatter(reduced_data_lle,"LLE",0.000001)




