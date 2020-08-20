import numpy as np
import matplotlib.pyplot as plt
import MNIST_tools
import Layer


MNIST_tools.downloadMNIST(path='MNIST_data', unzip=True)
x_train, y_train = MNIST_tools.loadMNIST(dataset="training", path="MNIST_data")
x_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path="MNIST_data")

onehot_y = Layer.one_hot(y_train,10)

#宣告 layer
layer_1 = Layer.Layer(784,256,"ReLu","h")
layer_2 = Layer.Layer(256,10,"Softmax","o")

dots = [i*100 for i in range(0,31)]
Accuracy = []
Loss = []

# a小題
for i in dots:
    # 拿出訓練好的參數
    layer_1.weight = np.load("p1_a_weight\\"+str(i)+"_layer1weight.npy")
    layer_2.weight = np.load("p1_a_weight\\"+str(i)+"_layer2weight.npy")

    # 計算
    layer_1.calculate(x_test)
    layer_2.calculate(layer_1.a)

    y_test = np.array(y_test)
    Accuracy.append(Layer.accuracy(layer_2.a,y_test))
    Loss.append(Layer.cross_entropy(layer_2.a, Layer.one_hot(y_test,10)))
    print(Accuracy)
    print(Loss)


plt.title('HW1_a')
plt.plot(Accuracy, color='green', label='accuracy')
plt.show()

plt.title('HW1_a')
plt.plot(Loss, color='red', label='Loss')
plt.show()














