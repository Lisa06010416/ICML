import numpy as np


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def differentionReLu(X):
    t = np.where(X <= 0, 0, 1)
    return t

def Predic(pre):
    return  pre.argmax(axis=1)

def accuracy(predic,groundtruth):
    predic = np.array(Predic(predic))
    true = ((predic - groundtruth) == 0).sum()
    return true/predic.size

def Loss(predic,groundtruth):
    predic = np.array(Predic(predic))
    true = ((predic - groundtruth) == 0).sum()
    false = predic.size-true
    return false/predic.size

def cross_entropy(X,Y):
    return -np.sum(np.log(X*Y+1e-9))/X.size

class Layer():
    def __init__(self, inputSize, outputSize, activateFun, layertype):
        # layertype
        self.layertype = layertype
        # activateFun
        self.activateFun = activateFun

        # output size
        self.outputSize = outputSize

        # input
        self.input = None
        self.inputWithBias = None
        self.datanum =None
        self.inputSize = inputSize

        # weight
        self.weight = self.randomlyGeneratedWeight()

        # calculateOutput
        self.z = None
        self.a = None

        # delta 紀錄
        self.delta = None

    # Update weight
    def updataWeight(self,l):
        # print(self.delta.T.shape)
        # print( self.inputWithBias.shape)
        # print(self.weight.shape)
        self.weight -= l * np.dot(self.inputWithBias.T, self.delta)/60000

    # calculate
    def calculate(self,input):
        # 更新 intput
        self.input = np.array(input)
        self.datanum, self.inputSize = input.shape
        self.inputWithBias = np.c_[self.input, np.ones(self.datanum)]

        #計算output
        self.z = self.calculateZ()
        self.a = self.calculateA()

        return self.a

    # 計算z
    def calculateZ(self):
        return np.dot(self.inputWithBias, self.weight)

    # 計算a
    def calculateA(self):
        if (self.activateFun == "ReLu"):
            return np.maximum(0, self.z)
        elif (self.activateFun == "Softmax"):
            return self.softmax(self.z)

    # normalize random 產生 weight 矩陣
    def randomlyGeneratedWeight(self):
        return np.random.randn(self.inputSize + 1, self.outputSize)

    def softmax(self, x):
        shift_x = x - np.max(x, 1).reshape(np.max(x, 1).shape[0], 1)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x, axis=1).reshape(np.max(x, 1).shape[0], 1)

    # print data
    def print(self):
        print("\n\n~~~~activateFun~~~~~")
        print(self.activateFun)

        print("\n\n~~~~~weight shape~~~~~: ")
        print(self.weight.shape)
        print("weight : ")
        print(self.weight)

        print("\n\n~~~~~input shape :~~~~~")
        print(self.input.shape)
        print("input :")
        print(self.input)
        print("inputWithBias shape:")
        print(self.inputWithBias.shape)
        print("inputWithBias :")
        print(self.inputWithBias)

        print("\n\n~~~~~Z shape :~~~~~")
        print(self.z.shape)
        print("Z:")
        print(self.z)

        print("\n\n~~~~~A shape :~~~~~")
        print(self.a.shape)
        print("A:")
        print(self.a)
