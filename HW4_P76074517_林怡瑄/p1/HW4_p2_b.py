import numpy as np
import Layer
import MNIST_tools

MNIST_tools.downloadMNIST(path='MNIST_data', unzip=True)
x_train, y_train = MNIST_tools.loadMNIST(dataset="training", path="MNIST_data")
x_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path="MNIST_data")

onehot_y = Layer.one_hot(y_train,10)

#宣告 layer
layer_1 = Layer.Layer(784,204,"ReLu","h")
layer_2 = Layer.Layer(204,202,"ReLu","h")
layer_3 = Layer.Layer(202,10,"Softmax","o")

# 紀錄一開始weight
np.save("p1_b_weight\\"+"0_layer1weight.npy", layer_1.weight)
np.save("p1_b_weight\\"+"0_layer2weight.npy", layer_2.weight)
np.save("p1_b_weight\\"+"0_layer3weight.npy", layer_3.weight)
for i in range(1000):
    
    print(i)
    # ~~~~~~~~~~~~~~~~~~~~~~~~forwordpass~~~~~~~~~~~~~~~~~~~~
    layer_1.calculate(x_train)
    # layer_1.print()
    layer_2.calculate(layer_1.a)
    # layer_2.print()
    layer_3.calculate(layer_2.a)

    #~~~~~~~~~~~~~~~~~~~~~~~backwordpass~~~~~~~~~~~~~~~~~~
    # 計算output layer 的 delta
    layer_3.delta = layer_3.a - onehot_y
    print("|layer_2.a - onehot_y|:")
    print(np.sum(np.abs(layer_3.delta)))

    # 計算layer 3 要到傳導回去的誤差
    bperr = np.dot(layer_3.delta, layer_3.weight.T)
    bperr = np.delete(bperr, -1, axis=1)  # bias的神經元不用導傳導  (60000,256)
    # 計算layer2 的delta
    layer_2.delta = bperr * Layer.differentionReLu(layer_2.z)

    # 計算layer 2 要到傳導回去的誤差
    bperr = np.dot(layer_2.delta, layer_2.weight.T)
    bperr = np.delete(bperr, -1, axis=1)  # bias的神經元不用導傳導  (60000,256)
    # 計算layer1 的delta
    layer_1.delta = bperr * Layer.differentionReLu(layer_1.z)

    #~~~~~~~~~~~~~~~~~~updataWeight~~~~~~~~~~~~~~~~~~~~~~~~~~~
    layer_1.updataWeight(0.000005)
    layer_2.updataWeight(0.000005)
    layer_3.updataWeight(0.000005)

    if (i+1)%10 == 0 :
        np.save("p1_b_weight\\"+str(i+1)+"_layer1weight.npy", layer_1.weight)
        np.save("p1_b_weight\\"+str(i+1) + "_layer2weight.npy", layer_2.weight)
        np.save("p1_b_weight\\" + str(i + 1) + "_layer3weight.npy", layer_3.weight)
