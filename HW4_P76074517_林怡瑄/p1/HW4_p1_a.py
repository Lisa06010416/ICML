import numpy as np
import Layer
import MNIST_tools

MNIST_tools.downloadMNIST(path='MNIST_data', unzip=True)
x_train, y_train = MNIST_tools.loadMNIST(dataset="training", path="MNIST_data")
x_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path="MNIST_data")

onehot_y = Layer.one_hot(y_train,10)



#宣告 layer
layer_1 = Layer.Layer(784,256,"ReLu","h")
layer_2 = Layer.Layer(256,10,"Softmax","o")

# 紀錄一開始weight
np.save("p1_a_weight\\"+"0_layer1weight.npy", layer_1.weight)
np.save("p1_a_weight\\"+"0_layer2weight.npy", layer_2.weight)

for i in range(10000):
    #~~~~~~~~~~~~~~~~~~~~~~~~forwordpass~~~~~~~~~~~~~~~~~~~~
    layer_1.calculate(x_train)
    layer_2.calculate(layer_1.a)

    #~~~~~~~~~~~~~~~~~~~~~~~backwordpass~~~~~~~~~~~~~~~~~~
    # 計算output layer 的 delta
    layer_2.delta = layer_2.a - onehot_y  #(60000,10)

    # 計算layer 2 要到傳導回去的誤差
    bperr = np.dot(layer_2.delta ,layer_2.weight.T) # (60000,257)
    bperr = np.delete(bperr,-1,axis=1)  # bias的神經元不用導傳導  (60000,256)

    # 計算layer1 的delta
    layer_1.delta = bperr*Layer.differentionReLu(layer_1.z)

    #~~~~~~~~~~~~~~~~~~updataWeight~~~~~~~~~~~~~~~~~~~~~~~~~~~
    layer_1.updataWeight(0.0005)
    layer_2.updataWeight(0.0005)
    
    print(i)
    print("|layer_2.a - onehot_y|:")
    print(np.sum(np.abs(layer_2.delta)))

    if (i+1)%100 == 0 :
        np.save("p1_a_weight\\"+str(i+1)+"_layer1weight.npy", layer_1.weight)
        np.save("p1_a_weight\\"+str(i+1) + "_layer2weight.npy", layer_2.weight)