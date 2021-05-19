## mnist 다운로드 및 데이터 추출
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

download_root = './MNIST_data'
mnistTrain = MNIST(download_root, train=True, download=True)
mnistTest = MNIST(download_root, train=False, download=True)

trainData = (np.asarray(mnistTrain.data) / 255)[:2000]
testData = (np.asarray(mnistTest.data) / 255)[:200]

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
    #return np.array ([1 if i >= 0 else 0.01 for i in x])
    return np.where(x <= 0, 0.01, 1)
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
## 컨볼루션 / 경사하강법

def conv2d(image, kernel, padding='zero'): # 들어온 커널에 맞게 image에 컨볼루션
    if padding=='zero':
        out_shape = image.shape
        image = np.pad(image, round((kernel.shape[0]-1)/2),'constant', constant_values=(0))
    else :
        out_shape = np.subtract(image.shape , kernel.shape)

    sub_matrices = np.lib.stride_tricks.as_strided(image,
                                                   shape=tuple(out_shape) + kernel.shape,
                                                   strides=image.strides * 2)
    return np.einsum('ij,klij->kl', kernel, sub_matrices)

def conv3d(image, kernel, padding='zero'):
    in_ch, out_ch = kernel.shape[0], kernel.shape[1]
    img_w, img_h = image.shape[-2], image.shape[-1]

    output = np.zeros([out_ch, img_w, img_h])

    for i in range(out_ch): #out ch
        for j in range(in_ch): #in ch
            # 단순히 conv2d를 input channel과 out channel 에 비례해 반복
            output[i] += conv2d(image[j], kernel[j,i], padding) 
    return output

def conv2d_back(image, filter, bias=0, stride=1, padding=0):
    h_filter, _ = filter.shape  # filter 차원만 가져오기

    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant')
    in_dim, _ = image.shape
    out_dim = int(((in_dim - h_filter) / stride) + 1)
    out = np.zeros((out_dim, out_dim))
    curr_y = out_y = 0

    while curr_y + h_filter <= in_dim:
        curr_x = out_x = 0

        while curr_x + h_filter <= in_dim:
            out[out_y, out_x] = (np.sum(filter * image[curr_y:curr_y + h_filter, curr_x:curr_x + h_filter]) + bias)/in_dim
            curr_x += stride
            out_x += 1
        curr_y += stride
        out_y += 1
    # in_dim 횟수만큼 누적했으니 다시 평균으로 나눠주어 return
    return out/in_dim

def conv3d_back(image, filter, bias=0, padding=0):
    ch = image.shape[0]
    output = np.zeros([ch,3,3])
    #print(output.shape)
    for i in range(ch):
        output[i] = conv2d_back(image[i], filter[i], bias=bias, padding=padding)
    return output

def pooling3d(image, poolsize=2):
    in_ch, size = image.shape[0], image.shape[1]
    output = np.zeros([in_ch, int(size/poolsize), int(size/poolsize)])
    index = np.zeros_like(image) # pooling 역전파시 위치 기억용
    for i in range(in_ch):
        row = 0
        for j in range(0,size,poolsize):
            col = 0
            for k in range(0,size,poolsize):
                output[i,row,col] = np.max(image[i,j:j+poolsize,k:k+poolsize])
                pooling_idx = np.asarray(np.unravel_index(image[i, j:j + poolsize, k:k + poolsize].argmax(),
                                       image[i, j:j + poolsize, k:k + poolsize].shape)) # 최대가 되는 좌표값만 지정
                pooling_idx += [j, k]
                index[i,pooling_idx[0], pooling_idx[1]] = 1 # 좌표값에 j,k를 더해 진짜 좌표값 더해서 해당 좌표에 1 대입
                col += 1
            row += 1
    return output, index


class NAG:
    def __init__(self):
        self.v = 0 #최초 속도 0으로 시작

    def calc(self, grad, lr = 0.01, m = 0.9):
        self.v = m * self.v - (lr * grad)
        return (m * self.v - lr * grad)

class ADAGRAD:
    def __init__(self):
        self.r = 0 #최초 0으로 시작

    def calc(self, grad, lr = 0.01):
        self.r = self.r + np.square(grad)
        return -(lr * ( 1 / np.sqrt(self.r+0.00001) ) * grad)

class RMSPROP:
    def __init__(self):
        self.r = 0 #최초 0으로 시작

    def calc(self, grad, lr = 0.01, a = 0.9):
        self.r = a * self.r + (1-a) * np.square(grad)
        return -(lr * ( 1 / np.sqrt(self.r+0.00001) ) * grad)

class ADAM:
    def __init__(self):
        self.v = 0 #최초 0으로 시작
        self.r = 0

    def calc(self, grad, lr = 0.01, a1 = 0.9, a2 = 0.999):
        self.r = a1 * self.r + (1-a1)*grad #속도
        r_hat = self.r / (1-a1)
        self.v = a2 * self.v + (1-a2)*grad*grad #그래디언트 누적
        v_hat = self.v / (1-a2)
        return -(lr * ( r_hat / (np.sqrt(v_hat)+0.0000001) ))

class BN:
    def __init__(self):
        self.gamma = 0.5
        self.beta = 0.1

    def bn_forward(self, input):
        mean = input.mean()
        var = input.var()

        self.std = np.sqrt(var + 0.000001)
        self.centered = input - mean
        self.normalized = self.centered / self.std
        out = self.gamma * self.normalized + self.beta

        return out

    def bn_backward(self, input, lr=0.01):
        size = input.shape

        dgamma = (input * self.normalized).sum()
        dbeta = input.sum()

        # 감마, beta 경사하강법으로 backprop
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

        dx_norm = input * self.gamma
        dx = 1 / size[1] / self.std * (size[1] * dx_norm -
                            dx_norm.sum(axis=0) -
                            self.normalized * (dx_norm * self.normalized).sum(axis=0))
        #return dx

## 네트워크 구성
class CNN:
    def __init__(self, ch1, nodes):
        self.nodes = nodes
        self.conv_layer1 = np.random.randn(1, ch1, 3, 3) #in_ch, out_ch, size
        self.bn_conv = BN()
        self.fcl1 = np.zeros([1,1]) # 입력 크기에 맞게 조절해야 함 -> forward부분에서 따로 생성
        self.bn_fcl = BN()
        self.outlayer = np.random.randn(nodes, 10)
        # 배치단위 error 누적용 params
        self.batch_grad_fcl = 0
        self.batch_grad_conv = 0
        self.batch_grad_out = 0
        self.batch_n = 0
        # 경사하강법 전용
        self.nag_out = NAG()
        self.nag_fcl = NAG()
        self.nag_conv = NAG()
        self.adgrd_out = ADAGRAD()
        self.adgrd_fcl = ADAGRAD()
        self.adgrd_conv = ADAGRAD()
        self.rms_out = RMSPROP()
        self.rms_fcl = RMSPROP()
        self.rms_conv = RMSPROP()
        self.adam_out = ADAM()
        self.adam_fcl = ADAM()
        self.adam_conv = ADAM()

    def forward(self, input):
        self.x = input
        self.out1 = leaky_relu(conv3d(input.reshape((1,) + input.shape), self.conv_layer1))  #[28,28]을 [1,28,28]로 확장하여 conv3d
        self.out1_pooling, self.out1_pool_index = pooling3d(self.out1)
        # 최초 1회 fully connected layer 초기화
        if self.fcl1.max() == 0:
            shape = self.out1_pooling.shape
            shape_mul = shape[0] * shape[1] * shape[2]
            self.fcl1 = np.random.randn(shape_mul, self.nodes)
        self.out1_bn = self.bn_conv.bn_forward(self.out1_pooling.reshape(1,-1))
        self.out2 = leaky_relu(self.out1_pooling.reshape(1,-1) @ self.fcl1)#[1*(7*7*ch2)]@[(7*7*ch2)*nodes]=[1,nodes]
        self.out2_bn = self.bn_fcl.bn_forward(self.out2)
        self.out3 = softmax(self.out2 @ self.outlayer) # [1,nodes]@[nodes,10] = [1,10]
        return self.out3

    def backward(self, ans):

        self.delta3 = self.out3 - ans
        self.grad_outlayer = self.out2.reshape(-1, 1) @ self.delta3.reshape(1, -1)  # [10, 1] @ [1, nodes] = [10, nodes]
        grad_out2 = self.delta3 @ self.outlayer.T  # 1,10 @ 10,nodes = [1, nodes]

        self.delta2 = grad_out2 * dleaky_relu(self.out2)
        self.bn_fcl.bn_backward(self.out1_pooling.reshape(-1, 1) @ self.delta2.reshape(1, -1)) # beta, gamma backprop

        self.grad_fcl1 = self.out1_pooling.reshape(-1,1) @ self.delta2.reshape(1,-1) # [(7*7*ch2), 1] @ [1, 10] = [(7*7*ch2), 10]
        self.bn_conv.bn_backward(self.delta2 @ self.fcl1.T)

        self.grad_out1 = self.delta2 @ self.fcl1.T # [1,10] @ [10, (7*7*ch2)] = [1, (7*7*ch2)]

        self.grad_out1 = self.grad_out1.reshape(self.out1_pooling.shape)
        self.grad_out1_unpool_mask = np.repeat(np.repeat(self.grad_out1, 2, axis=1), 2, axis=2) # unpooling 실시
        self.grad_out1_unpool = self.grad_out1_unpool_mask * self.out1_pool_index # pooling 된 곳 추적해 그곳만 grad로 사용

        delta1 = self.grad_out1_unpool * dleaky_relu(self.out1)
        self.grad_conv = conv3d_back(self.out1, delta1, padding=1) # 순전파시에 패딩을 했기 때문에 1칸만큼 다시 패딩

        # 배치단위 누적
        self.batch_grad_out += self.grad_outlayer
        self.batch_grad_fcl += self.grad_fcl1
        self.batch_grad_conv += self.grad_conv
        self.batch_n += 1

    def update(self, lr=0.001):
        self.outlayer -= lr * (self.batch_grad_out / self.batch_n)
        self.fcl1 -= lr * (self.batch_grad_fcl/self.batch_n)
        self.conv_layer1[0] -= lr * (self.batch_grad_conv/self.batch_n)
        # update 후에는 누적된 값들 초기화
        self.batch_grad_out, self.batch_grad_conv, self.batch_grad_fcl, self.batch_n = 0,0,0,0

    def update_nag(self, lr=0.001):
        self.outlayer += self.nag_out.calc(self.batch_grad_out / self.batch_n, lr=lr)
        self.fcl1 += self.nag_fcl.calc(self.batch_grad_fcl/self.batch_n, lr=lr)
        self.conv_layer1[0] += self.nag_conv.calc(self.batch_grad_conv/self.batch_n, lr=lr)
        # update 후에는 누적된 값들 초기화
        self.batch_grad_out, self.batch_grad_conv, self.batch_grad_fcl, self.batch_n = 0,0,0,0

    def update_adgrd(self, lr=0.001):
        self.outlayer += self.adgrd_out.calc(self.batch_grad_out / self.batch_n, lr=lr)
        self.fcl1 += self.adgrd_fcl.calc(self.batch_grad_fcl/self.batch_n, lr=lr)
        self.conv_layer1[0] += self.adgrd_conv.calc(self.batch_grad_conv/self.batch_n, lr=lr)
        # update 후에는 누적된 값들 초기화
        self.batch_grad_out, self.batch_grad_conv, self.batch_grad_fcl, self.batch_n = 0,0,0,0

    def update_rms(self, lr=0.001):
        self.outlayer += self.rms_out.calc(self.batch_grad_out / self.batch_n, lr=lr)
        self.fcl1 += self.rms_fcl.calc(self.batch_grad_fcl/self.batch_n, lr=lr)
        self.conv_layer1[0] += self.rms_conv.calc(self.batch_grad_conv/self.batch_n, lr=lr)
        # update 후에는 누적된 값들 초기화
        self.batch_grad_out, self.batch_grad_conv, self.batch_grad_fcl, self.batch_n = 0,0,0,0

    def update_adam(self, lr=0.001):
        self.outlayer += self.adam_out.calc(self.batch_grad_out / self.batch_n, lr=lr)
        self.fcl1 += self.adam_fcl.calc(self.batch_grad_fcl/self.batch_n, lr=lr)
        self.conv_layer1[0] += self.adam_conv.calc(self.batch_grad_conv/self.batch_n, lr=lr)
        # update 후에는 누적된 값들 초기화
        self.batch_grad_out, self.batch_grad_conv, self.batch_grad_fcl, self.batch_n = 0,0,0,0
##
model = CNN(5,200)
max_epoch=15
batchSize = 50
batch_for_epoch = (int)(len(trainData)/batchSize)
##
accu_b, loss_b = np.zeros([max_epoch*batch_for_epoch]), np.zeros([max_epoch*batch_for_epoch])
accu, loss = np.zeros([max_epoch]), np.zeros([max_epoch])
for epoch in range(max_epoch):
    # 순서 섞기
    shfl = np.arange(len(trainData))
    np.random.shuffle(shfl)
    trainData = trainData[shfl, :]
    trainLabel = trainLabel[shfl, :]
    # 배치단위로 순전파 및 역전파
    for batch in range(batch_for_epoch):
        l = 0
        b_idx = epoch*batch_for_epoch+batch # 배치단위 인덱스

        for iteration in range(batchSize):
            idx = (batch * batchSize) + iteration # 현재 배치에서 위치 인덱스
            pred = model.forward(trainData[idx])
            model.backward(trainLabel[idx])
            # 로스 및 정확도 누적
            loss_b[b_idx] += np.sum((pred - trainLabel[idx]) ** 2)
            if pred.argmax() == trainLabel[idx].argmax():
                accu_b[b_idx] += 1

        # 1개 배치 사이즈만큼 순전파를 했다면, 누적된 error 토대로 update
        model.update_adam(lr=0.01)
        # loss 및 정확도 계산
        loss_b[b_idx] /= batchSize
        accu_b[b_idx] /= batchSize
        print("loss:{}, accu:{}, progress:{}/{}".format(loss_b[b_idx], accu_b[b_idx], b_idx,max_epoch*batch_for_epoch))
    loss[epoch] = loss_b[epoch*batch_for_epoch : epoch*batch_for_epoch+batch].mean()
    accu[epoch] = accu_b[epoch * batch_for_epoch: epoch * batch_for_epoch + batch].mean()
## plot
plt.plot(range(int(batch_for_epoch/2), len(loss_b)+1, batch_for_epoch), loss, c='blue')
plt.plot(range(int(batch_for_epoch/2), len(accu_b)+1, batch_for_epoch), accu, c='red')
plt.scatter(range(len(loss_b)),loss_b, s=2, c='skyblue')
plt.scatter(range(len(accu_b)),accu_b, s=2, c='orange')
plt.legend(["Loss", "accuracy"])
plt.xlabel('iteration')
plt.grid(1)
plt.show()
## test
testacc=0
for i in range(len(testData)):
    test_pred = model.forward(testData[i])
    if test_pred.argmax() == testLabel[i].argmax():
        testacc += 1
print(testacc/i)
print("train:",accu[-1])
##

