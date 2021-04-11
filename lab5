## mnist 다운로드 및 데이터 추출
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

download_root = './MNIST_data'
mnistTrain = MNIST(download_root, train=True, download=True)
mnistTest = MNIST(download_root, train=False, download=True)

trainData = (np.asarray(mnistTrain.data) / 255)
testData = (np.asarray(mnistTest.data) / 255)

#one hot encoding
trainLabel = np.zeros([len(trainData), 10], dtype=int)
testLabel = np.zeros([len(trainData), 10], dtype=int)

for i in range(len(trainData)):
    trainLabel[i, np.asarray(mnistTrain.targets)[i]] = 1
for i in range(len(testData)):
    testLabel[i, np.asarray(mnistTest.targets)[i]] = 1


## 컨볼루션 설정

def conv2d(image, kernel, padding='zero'):

    if padding=='zero':
        out_shape = image.shape
        image = np.pad(image, round((kernel.shape[0]-1)/2),'constant', constant_values=(0))
    else :
        out_shape = np.subtract(image.shape , kernel.shape)

    sub_matrices = np.lib.stride_tricks.as_strided(image,
                                                   shape=tuple(out_shape) + kernel.shape,
                                                   strides=image.strides * 2)
    return np.einsum('ij,klij->kl', kernel, sub_matrices)

## ReLU 및 Sigmoid 구성
def leaky_relu(x):
    return np.maximum(0.01*x,x)

def dleaky_relu(x):
    return np.array ([1 if i >= 0 else 0.01 for i in x])
#
def ReLU(x):
    return np.maximum(0,x)

def dReLU(x):
    return (x > 0).astype(np.int)
#
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
#
def tanh(x):
    return (2 * sigmoid(x) - 1)

def dtanh(x):
    return ((1 - tanh(x)**2) / 2)
#
def softmax(x) :
    exp_x = np.exp(x - np.max(x)) # overflow 방지
    return (exp_x / np.sum(exp_x))

## 네트워크 구성
class CNN:
    def __init__(self, ch1, ch2):
        self.conv_layer1 = np.random.randn(1, ch1, 3, 3) #in_ch, out_ch, size
        self.conv_layer1_b = np.random.randn(1, ch1)

        self.conv_layer2 = np.random.randn(ch1, ch2, 3, 3)
        self.conv_layer2_b = np.random.randn(1, ch2)

        self.fcl = np.random.randn(ch2 * 784,10)

    def forward(self, input):
        self.x = input
        self.out1 = np.zeros([1,self.conv_layer1.shape[1], input.shape[0], input.shape[1]])
        for i in range(self.conv_layer1.shape[1]):
            self.out1[0,i] = leaky_relu(conv2d(input, self.conv_layer1[0,i,:,:]))# + self.conv_layer1_b[0,i])

        self.out2 = np.zeros([1,self.conv_layer2.shape[1], input.shape[0], input.shape[1]])
        for i in range(self.conv_layer2.shape[1]):
            for j in range(self.conv_layer2.shape[0]):
                self.out2[0,i] += conv2d(self.out1[0,j,:,:], self.conv_layer2[j,i,:,:])
            self.out2[0,i] = leaky_relu(self.out2[0,i])# + self.conv_layer2_b[0,i])

        self.out3 = softmax(self.out2.reshape(1,-1) @ self.fcl)#[1*15000]@[15000*10]=[1,10]

        return self.out3

    def backward(self, ans):
        delta3 = self.out3 - ans
        self.grad_layer3 = self.out2.reshape(-1,1) @ delta3.reshape(1,-1)# [10, 1] @ [1, nodes] = [10, nodes]

        grad_out2 = delta3.T @ self.layer3.T # [1,10] @ [10, nodes] = [1, nodes]

        delta2 = grad_out2 * dleaky_relu(self.out2) # delta2 = [1, nodes] * [1, nodes] = [1, nodes]
        self.grad_layer2 = self.out1.reshape(-1,1) @ delta2.reshape(1,-1) # [nodes,1] @ [1,nodes] = [nodes,nodes]
        self.grad_layer2_b = delta2

        grad_out1 = delta2.T @ self.layer2.T # [1,10] @ [10, nodes] = [1, nodes]

        delta1 = grad_out1 * dleaky_relu(self.out1) # [1, nodes] 원소별 곱 시행
        self.grad_layer1 = self.x.reshape(-1,1) @ delta1.reshape(1, -1)  # [nodes,1] @ [1,784] = [nodes,784]
        self.grad_layer1_b = delta1
##
model = CNN(10, 20)
output = model.forward(trainData[0,:,:])

##

