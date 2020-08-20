# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:58:13 2018

@author: Lisa
"""
import cv2
import numpy as np
from os import listdir
from os.path import join
from sklearn.decomposition import PCA
import math

def PSNR(img1, img2):
    mse = np.mean((img1-img2)** 2 )
    return 20 * math.log10(255 / math.sqrt(mse))



mypath = "training.db"
files = listdir(mypath)
h=128
l=128
images = []
for f in files:
    fullpath = join(mypath, f)
    gray = cv2.imread(fullpath,cv2.IMREAD_GRAYSCALE).flatten()
    images.append(gray)

    
#meanface
meanface=np.array(images).mean(axis=0)
meanface = cv2.normalize(meanface, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imwrite('meanface.jpg', meanface.reshape(128,128))
meanface = meanface.reshape(1,16384)[0]

#Eigenfaces = []
#for i in range(len(images)):
#    t = images[i]-meanface
#    Eigenfaces.append(images[i])#-meanface)
#    cv2.imshow('Eigenfaces[i]' ,t.reshape(128,128))
#    cv2.waitKey(0)
#~~~~~~~~~~~~~~1~~~~~~~~~~~~~~~~~~~~~~~~~
# Show the mean (average) face
print("meanface")
cv2.imshow('meanface',meanface.reshape(128,128))
cv2.waitKey(0)
cv2.destroyAllWindows()
# top 5 eigenfaces and their corresponding eigenvalues in a descending order.
pca=PCA(n_components=5) 
pca.fit(images)

print("1. top 5 eigenfaces and their corresponding eigenvalues in a descending order.")
for i in range(5):
    print(pca.explained_variance_ratio_[i])
    t = cv2.normalize(pca.components_[i], None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imwrite('TOP'+str(i)+".jpg", t.reshape(128,128))
    cv2.imshow('TOP'+str(i),t.reshape(128,128))
    cv2.waitKey(0)
cv2.destroyAllWindows()
    
#~~~~~~~~~~~~~~2~~~~~~~~~~~~~~~~~~~~~~~~~~
print("2. Given a test image (hw03-test.tif, or you can use your own image), compute the top 10 eigenface coefficients.")
testface = cv2.imread("hw03-test.tif",cv2.IMREAD_GRAYSCALE).flatten()
#testface = testface_org-meanface

pca=PCA(n_components=10)
pca.fit(images)
project=pca.transform(testface.reshape(1, -1))
print (project) 


#~~~~~~~~~~~~~3~~~~~~~~~~~~~~~~~~~~~~~~~~
print("3. Keep only first K (K=5,10,15,20, and 25) coefficients and use them to reconstruct the image in the pixel domain. Compare the reconstructed image with the original image by PSNR value.")
for i in [5, 10, 15, 20, 25]:
    pca=PCA(n_components=i)
    pca.fit(images)   
    project=pca.transform(testface.reshape(1, -1))
    reconstructImg = np.dot(project,pca.components_) 
    reconstructImg = reconstructImg+ meanface
    
    reconstructImg = cv2.normalize(reconstructImg, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    psnr = PSNR(testface,reconstructImg)
    cv2.imshow('reconstructImg '+str(i),reconstructImg.reshape(128,128))
    cv2.waitKey(0)
    cv2.imwrite('reconstructImg '+str(i)+".jpg", reconstructImg.reshape(128,128)) 
    
    print("psnr : "+str(psnr))
    
cv2.destroyAllWindows()
    




