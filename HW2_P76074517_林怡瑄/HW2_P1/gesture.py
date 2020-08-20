from skimage import exposure    
from skimage import feature
import cv2
from os import listdir
from os.path import isfile, isdir, join
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

def readfile(filepath):
   imageList=[]
   features=[]
   targes = []
   files = listdir(filepath)
   for f in files:
       fullpath = join(filepath, f)
       image = cv2.imread(fullpath)
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       imageList.append(gray)
       H = feature.hog(gray, orientations=4, pixels_per_cell=(100, 100),cells_per_block=(1,1), transform_sqrt=True,feature_vector=True)
       features.append(H)
       targes.append(f[0])
   return features,targes

#def getFeature_moments(imageList):
#   feature=[]
#   for i in imageList:
#       feature.append(cv2.HuMoments(cv2.moments(i)).flatten())
#   return feature

target_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
features,targes = readfile("training/")
features_t,targes_t = readfile("test/")
scoring = ['precision_macro', 'recall_macro']

print("~~~~~~~~~~~~~~~linear kernel~~~~~~~~~~~~~~~~~")
clf = svm.SVC(kernel='linear')
scores = cross_validate(clf, features,targes, scoring=scoring, cv=5, return_train_score=False)
print("cross_validation scores")
print(scores)

print("~~~precision validate~~~")
sum = 0
for i in scores['test_precision_macro']:
    sum+=i
    print(i,end=' ')
    
print("\n~~~precision validate mean~~~")
print(sum/5)

print("~~~recall validate~~~")
sum = 0
for i in scores['test_recall_macro']:
    sum+=i
    print(i,end=' ')  
    
print("\n~~~recall validate mean~~~")
print(sum/5)

clf.fit(features,targes)
result = clf.predict(features_t)
print(classification_report(targes_t, result, target_names=target_names))

print("~~~~~~~~~~~~~~~Rbf kernel~~~~~~~~~~~~~~~~~")
clf2 = svm.SVC(kernel = "rbf")
scores = cross_validate(clf2, features,targes, scoring=scoring, cv=5, return_train_score=False)
print("cross_validation scores")
print(scores)

print("~~~precision validate~~~")
sum = 0
for i in scores['test_precision_macro']:
    sum+=i
    print(i,end=' ')
    
print("\n~~~precision validate mean~~~")
print(sum/5)

print("~~~recall validate~~~")
sum = 0
for i in scores['test_recall_macro']:
    sum+=i
    print(i,end=' ')  
    
print("\n~~~recall validate mean~~~")
print(sum/5)

clf2.fit(features,targes)
result2 = clf2.predict(features_t)
print(classification_report(targes_t, result2, target_names=target_names))