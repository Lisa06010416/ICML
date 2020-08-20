import numpy as np

target = []
point = []
b = 0
C = 1
Tol = 0.001
n = 0  #樣本數
kernelMat = []

        
def readfil():
    with open('liver-disorders_scale.txt','r') as f :
        for i in f.readlines():
            splitdata = i.split()
            temp=[]
            for k in splitdata:
                try:
                    temp.append(float(k.split(':')[1]))
                except:
                    if splitdata[0]=='0':
                        target.append(-1)
                    else:
                        target.append(1)
            point.append(temp)
def L_H_bound(a1,a2,y1,y2):
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~L_H_bound~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if y1 != y2:
        L = max(0, a2-a1)
        H = min(C, C+a2-a1)
    else :
        L = max(0, a1+a2-C)
        H = min(C, a1+a2)
    return L,H

    
def calcKernelMatrix():
    kernelMat = np.array(np.zeros((n, n)))
    for i in range(n):
        kernelMat[:, i] = calcKernelValue(point, point[i, :])
    return kernelMat


def calcKernelValue(x, xi):
    kernelValue = np.array(np.zeros((n, 1)))
    kernelValue = np.dot(x , xi)
    return kernelValue

def calcError(i):
    f = np.dot(a*target,kernelMat[:, i]) + b    
    return (f - target[i])
def preTarget(i):
    f = np.dot(a*target,kernelMat[:, i]) + b
    return f
def takeStep(i1,i2):

    global b,a
    a=list(a)
    if i1 == i2:
        return 0
           
    alph1 = a[i1]
    alph2 = a[i2]
    y1 = target[i1]
    y2 = target[i2]
    E1 = calcError(i1)
    E2 = calcError(i2)
    s = y1*y2
    
    L,H = L_H_bound(alph1,alph2,y1,y2)
    if (L==H):
        return 0
    k11 = kernelMat[i1, i1]
    k12 = kernelMat[i1, i2]
    k22 = kernelMat[i2, i2]
    eta = k11 + k22 -2*k12

    if(eta > 0):
        a2 = alph2 + y2*(E1-E2)/eta
        if a2 < L:
            a2 = L
        elif (a2>H):
            a2 = H
    else:
        return 0

    if (np.abs(a2-alph2) < Tol*(a2+alph2+Tol)):
        return 0
    a1 = alph1 + s*(alph2-a2)

    # --------------------updata b-------------
    b1 = E1+y1*(a1-alph1)*k11+y2*(a2-alph2)*k12+b
    b2 = E2+y1*(a1-alph1)*k12+y2*(a2-alph2)*k22+b


    if(a1 > 0) and (a1 < C):
        b=b1
    elif(a2 > 0) and (a2 < C):
        b=b2
    else:
        b=(b1+b2)/2

    # --------------------updata a-------------  
    # Store a1 in the alpha array
    a[i1] = a1
    # Store a2 in the alpha array
    a[i2] = a2     
    return 1

def Heuristic(E2):
    max=-9999999
    maxIndex = -1
    for i in range(n):
        
        E1 = calcError(i)
        if(np.abs(E1-E2)>max):
            max = np.abs(E1-E2)
            maxIndex = i
    return maxIndex



def  examineExample(i2):
    
    y2 = target[i2]
    alpha2 = a[i2]
    E2 = calcError(i2)        
    r2 = E2*y2
    
    if ( r2 < -Tol and alpha2 < C) or  ( r2 > Tol and alpha2 > 0):  #KKT
        i1 = Heuristic(E2)
        if takeStep(i1, i2):
            return 1
        for i in range(len(a)):
            i1 = i
            if takeStep(i1, i2):
                return 1
    return 0

def init():
    global  target,point,n,errorCache,kernelMat
    readfil()
    n=len(target)
    target = np.array(target)
    point = np.array(point)
    errorCache = np.array(np.zeros((n, 2)))
    kernelMat = calcKernelMatrix()
    
    
def predict():
    count = 0
    predic = []
    for i in range(n):
        pre = 0
        if preTarget(i) > 0:
            pre = 1
        elif preTarget(i) < 0:
            pre = -1
        predic.append(pre)
        if target[i] == pre:
            count+=1
    print("accuracy : ",end="")
    print(count/n)
#    print("predic : ")
#    print(target)
#    print (predic)
    print("a :")
    print(a)

    
if __name__ == '__main__':
    
    readfil()  #讀檔
    init()  #初始化
    a = np.array([0 for i in range(n)])
    

    numChanged = 1
    examineAll = 1
    while (numChanged > 0 or examineAll==1):
         numChanged = 0
         
         for I in range(n):
             numChanged += examineExample(I)
         if (examineAll):
             for I in range(n):# loop I over all training examples
                 numChanged += examineExample(I)
         else:
             #loop I over examples where alpha is not 0 & not C
             for I in range(n):
                 if a[I] != 0 and a[I] != C:
                     numChanged += examineExample(I)
         predict()
         print(numChanged)
         if (examineAll == 1):
             examineAll = 0
         elif(numChanged==0):
             examineAll = 1