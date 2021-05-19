

## 데이터 뽑아서 앞 2종류만 선택
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

irisDataSet = load_iris()
# 클래스 정보를 뽑아와서 클래스 인덱스 저장
irisClass = irisDataSet.target
class2Index = np.asarray(np.where(irisClass == 0))
class1Index = np.asarray(np.where(irisClass == 1))

# 데이터 추출
iris1 = irisDataSet.data[class1Index,:].squeeze()
iris2 = irisDataSet.data[class2Index,:].squeeze()

## 학습/ 테스트 데이터 분류
trainSize = 40
testSize = 10


#5번째 열은 레이블( 0 혹은 1)
train = np.zeros([2 * trainSize, 5])
train[:trainSize, 0:4] = iris1[:trainSize,:]
train[trainSize:, 0:4] = iris2[:trainSize,:]
train[:trainSize, 4] = -1
train[trainSize:, 4] = 1

test = np.zeros([2 * testSize, 5])
test[:testSize, 0:4] = iris1[trainSize:,:]
test[testSize:, 0:4] = iris2[trainSize:,:]
test[:testSize, 4] = -1
test[testSize:, 4] = 1

## 이진 분류기 설계(CLASS)
class classifier:
    def __init__(self, size):
        self.w = np.random.rand(size,1)
        self.b = np.random.rand(1)

    def forward(self, input):
        hypothesis = input @ self.w + self.b
        y = 1 if hypothesis > 0 else -1
        return y

    def backward(self, ans, x, lr=0.001):
        for k in range(len(x)):
            self.w[k] += lr * ans * x[k]
        self.b += lr * ans

## 분류기 시행
lr = 0.00005
max_epoch = 200

model = classifier(4)

accuracy= np.zeros([max_epoch])
for epoch in range(max_epoch):
    accu = 0.0
    np.random.shuffle(train)
    for i in range(len(train)): # stocastic

        y = model.forward(train[i, 0:4])

        if y != train[i, 4]: # 틀렸다면 경사하강법을 통한 가중치 업데이트
            model.backward(train[i, 4], train[i, 0:4], lr)
        else: accu += 1 # 맞았다면 정확도++

    accuracy[epoch] = accu / len(train)
    if epoch % 10 == 0: print("Epoch : {}, accuracy : {}".format(epoch, accuracy[epoch]))

## 그래프 띄우기
plt.plot(range(max_epoch), accuracy, label='accuracy')
plt.xlabel("epoch"); plt.legend(loc = 'lower right'); plt.axis([0,max_epoch,0,1.1]); plt.grid(1)

## test 데이터에 대해 해보기

accu= 0
for i in range(len(test)):  # stocastic

    y = model.forward(test[i, 0:4])
    if y == test[i, 4]: accu += 1

accu /= len(test)
print("test accuracy : {}".format(accu))

##


##

