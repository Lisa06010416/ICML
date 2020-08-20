import numpy as np
import Layer
import MNIST_tools
import matplotlib.pyplot as plt

MNIST_tools.downloadMNIST(path='MNIST_data', unzip=True)
x_train, y_train = MNIST_tools.loadMNIST(dataset="training", path="MNIST_data")
x_test, y_test = MNIST_tools.loadMNIST(dataset="testing", path="MNIST_data")

onehot_y = Layer.one_hot(y_train,10)

#宣告 layer
layer_1 = Layer.Layer(784,128,"ReLu","h")
layer_2 = Layer.Layer(128,784,"Sigmoid","o")

#banch
B = 200
for i in range(10000):

    print(i)
    for banch in range(0,int(len(x_train)/B)):
        B_x_train = x_train[banch*B:banch*(B)+B]
        #~~~~~~~~~~~~~~~~~~~~~~~~forwordpass~~~~~~~~~~~~~~~~~~~~
        layer_1.calculate(Layer.normalize(B_x_train))
        layer_2.calculate(layer_1.a)

        #~~~~~~~~~~~~~~~~~~~~~~~backwordpass~~~~~~~~~~~~~~~~~~
        # 計算output layer 的 delta
        layer_2.delta = (layer_2.a - np.array(Layer.normalize(B_x_train)))*(1-layer_2.a )*(layer_2.a)

        # 計算layer 2 要到傳導回去的誤差
        bperr = np.dot(layer_2.delta ,layer_2.weight.T)  # (60000,257)
        bperr = np.delete(bperr,-1,axis=1)  # bias的神經元不用導傳導  (60000,256)
        # 計算layer1 的delta
        layer_1.delta = bperr*Layer.differentionReLu(layer_1.z)

        #~~~~~~~~~~~~~~~~~~updataWeight~~~~~~~~~~~~~~~~~~~~~~~~~~~
        layer_1.updataWeight(0.00005)
        layer_2.updataWeight(0.00005)

        print("|layer_2.delta|:")
        print(np.sum(np.abs(layer_2.a - np.array(B_x_train/256))))

    # 紀錄 weight
    if (i+1)%10 == 0:
        np.save("p2_weight\\"+str(i+1)+"_layer1weight.npy", layer_1.weight)
        np.save("p2_weight\\"+str(i+1) + "_layer2weight.npy", layer_2.weight)

        print("[Visualize Training Data]")
        plt.imshow(x_train[0].reshape([28,28]), cmap="gray")
        plt.savefig("p2_feature\\"+str(i+1)+"groundtruth.png")
        plt.show()
        plt.imshow(layer_2.a[0].reshape([28,28]), cmap="gray")
        plt.savefig("p2_feature\\"+str(i+1)+"predict.png")
        plt.show()