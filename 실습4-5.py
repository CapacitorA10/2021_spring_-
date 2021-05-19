## mnist 다운로드 및 데이터 추출
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

download_root = './MNIST_data'
mnistTrain = MNIST(download_root, train=True, download=True)
mnistTest = MNIST(download_root, train=False, download=True)

trainData = (np.asarray(mnistTrain.data) / 255)
testData = (np.asarray(mnistTest.data) / 255)

## 784*1 벡터 변환 / one hot encoding
trainData = trainData.reshape([-1, 784])
testData = testData.reshape([-1, 784])
# 벡터 변환 끝


#one hot encoding
trainLabel = np.zeros([len(trainData), 10], dtype=int)
testLabel = np.zeros([len(trainData), 10], dtype=int)

for i in range(len(trainData)):
    trainLabel[i, np.asarray(mnistTrain.targets)[i]] = 1
for i in range(len(testData)):
    testLabel[i, np.asarray(mnistTest.targets)[i]] = 1

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
class mlp:
    def __init__(self, nodes):
        self.layer1 = np.random.rand(784, nodes)
        self.layer1_b = np.random.rand(nodes)

        self.layer2 = np.random.rand(nodes, 10)
        self.layer2_b = np.random.rand(10)

    def forward(self, input):
        self.x = input
        self.out1 = ReLU(self.x @ self.layer1 + self.layer1_b) # [1,784] @ [784,nodes] = [1,nodes]
        self.out2 = softmax(self.out1 @ self.layer2 + self.layer2_b) # [1, nodes] @ [nodes, 10] = [1,10]
        return self.out2

    def backward(self, ans):
        delta2 = self.out2 - ans #행렬곱이 아닌 원소별 곱을 해야 [10, 1] 행렬로 출력됨
        self.grad_layer2 = self.out1.reshape(-1,1) @ delta2.reshape(1,-1)
        self.grad_layer2_b = delta2

        grad_out1 = delta2.T @ self.layer2.T # [1,10] @ [10, nodes] = [1, nodes]

        delta1 = grad_out1 * dReLU(self.out1) # [1, nodes] 원소별 곱 시행
        self.grad_layer1 = self.x.reshape(-1,1) @ delta1.reshape(1,-1)
        self.grad_layer1_b = delta1

    def update(self, lr=0.001):
        self.layer2 += -lr * self.grad_layer2
        self.layer2_b += -lr * self.grad_layer2_b
        self.layer1 += -lr * self.grad_layer1
        self.layer1_b += -lr * self.grad_layer1_b

## 네트워크 구성
class mlp2:
    def __init__(self, nodes):
        self.layer1 = np.random.rand(784, nodes)
        self.layer1_b = np.random.rand(nodes)

        self.layer2 = np.random.rand(nodes, nodes)
        self.layer2_b = np.random.rand(nodes)

        self.layer3 = np.random.rand(nodes, 10)
        self.layer3_b = np.random.rand(10)

    def forward(self, input):
        self.x = input
        self.out1 = leaky_relu(self.x @ self.layer1+ self.layer1_b) # [1,784] @ [784,nodes] = [1,nodes]
        self.out2 = leaky_relu(self.out1 @ self.layer2+ self.layer2_b)# [1,nodes] @ [nodes,nodes] = [1,nodes]
        self.out3 = softmax(self.out2 @ self.layer3 + self.layer3_b) # [1, nodes] @ [nodes, 10] = [1,10]
        return self.out3

    def backward(self, ans):
        delta3 = self.out3 - ans
        self.grad_layer3 = self.out2.reshape(-1,1) @ delta3.reshape(1,-1)# [10, 1] @ [1, nodes] = [10, nodes]
        self.grad_layer3_b = delta3

        grad_out2 = delta3.T @ self.layer3.T # [1,10] @ [10, nodes] = [1, nodes]

        delta2 = grad_out2 * dleaky_relu(self.out2) # delta2 = [1, nodes] * [1, nodes] = [1, nodes]
        self.grad_layer2 = self.out1.reshape(-1,1) @ delta2.reshape(1,-1) # [nodes,1] @ [1,nodes] = [nodes,nodes]
        self.grad_layer2_b = delta2

        grad_out1 = delta2.T @ self.layer2.T # [1,10] @ [10, nodes] = [1, nodes]

        delta1 = grad_out1 * dleaky_relu(self.out1) # [1, nodes] 원소별 곱 시행
        self.grad_layer1 = self.x.reshape(-1,1) @ delta1.reshape(1, -1)  # [nodes,1] @ [1,784] = [nodes,784]
        self.grad_layer1_b = delta1

    def update(self, lr=0.001):
        self.layer3 += -lr * self.grad_layer3
        self.layer3_b += -lr * self.grad_layer3_b
        self.layer2 += -lr * self.grad_layer2
        self.layer2_b += -lr * self.grad_layer2_b
        self.layer1 += -lr * self.grad_layer1
        self.layer1_b += -lr * self.grad_layer1_b

##초기값 설정
node_num = 30

## 실행
model = mlp(node_num)
max_epoch = 30
accuracy, loss = np.zeros([max_epoch*10]), np.zeros([max_epoch*10])
accuracy_t, loss_t = np.zeros([max_epoch]), np.zeros([max_epoch])
for epoch in range(max_epoch):
    accu, accu_t = 0.0, 0.0
    l, l_t = 0,0
    # 순서 섞기
    shfl = np.arange(len(trainData))
    np.random.shuffle(shfl)
    trainData = trainData[shfl,:]
    trainLabel = trainLabel[shfl,:]

    for i in range(len(trainData)):  # stocastic
        y = model.forward(trainData[i,:])
        model.backward(trainLabel[i,:])
        model.update(lr = 0.001)


        # 학습완료. 밑으로는 정확도/로스 측정 및 데이터 저장
        l += np.sum((y - trainLabel[i,:])**2)
        ymax = np.where(y == np.max(y))
        labelMax = np.where(trainLabel[i,:] == np.max(trainLabel[i,:]))
        accu += (ymax == labelMax)

        if (i%5999 == 0) & (i!=0):
            iter = int(i/5999)-1
            accu /= 6000
            l /= 6000
            print("Epoch : {}, iter : {}/9, loss : {}, Accu : {}".format(epoch, iter, l, accu))
            accuracy[epoch*10+iter] = accu
            loss[epoch*10+iter] = l


    # Test데이터 로스 정확도 비교
    for i in range(len(testData)):
        # 추론 진행
        y = model.forward(testData[i, :])

        l_t += np.sum((y - testLabel[i, :]) ** 2)
        ymax = np.where(y == np.max(y))
        labelMax = np.where(testLabel[i, :] == np.max(testLabel[i, :]))
        accu_t += (ymax == labelMax)
    accuracy_t[epoch] = accu_t/i
    loss_t[epoch] = l_t/i


## 그래프 그리기
plt.plot(np.linspace(0,29.9,300), accuracy)
plt.plot(np.linspace(0,29.9,300), loss)
plt.plot(np.linspace(0,29.9,300), np.repeat(accuracy_t,10))
plt.plot(np.linspace(0,29.9,300), np.repeat(loss_t,10))
plt.legend(["Train Acc", "Train Loss", "Test Acc", "Test Loss"])
plt.grid(1)
plt.axis([-1,31 , 0, 1.2])