
##Lab 11 MNIST and Convolutional Neural Network
import torch
import torch.nn.init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = 'cuda'

learning_rate = 0.001
max_epoch = 20
batch_size = 100


## data load
def loadTraffic(type = 'train'): # 주어진 data를 numpy로 불러오기 + Tensor type으로 변환까지
    if type == 'train':
        name = np.loadtxt('TrafficSignDB/TsignRecgTrain4170Annotation.txt', delimiter=';', dtype='str',usecols=[0])
        labels = np.loadtxt('TrafficSignDB/TsignRecgTrain4170Annotation.txt', delimiter=';', dtype='int', usecols=[1,2,3,4,5,6,7])
        data = {}
        for i in range(len(name)):
            data[i] = torch.from_numpy(plt.imread('TrafficSignDB/tsrd-train/{}'.format(name[i])))
    elif type == 'test':
        name = np.loadtxt('TrafficSignDB/TsignRecgTest1994Annotation.txt', delimiter=';', dtype='str',usecols=[0])
        labels = np.loadtxt('TrafficSignDB/TsignRecgTest1994Annotation.txt', delimiter=';', dtype='int', usecols=[1,2,3,4,5,6,7])
        data = {}
        for i in range(len(name)):
            data[i] = torch.from_numpy(plt.imread('TrafficSignDB/TSRD-Test/{}'.format(name[i])))
    else: print('load error'); exit(7)
    return data, torch.from_numpy(labels).long() #long type으로 label을 반납해야 closs entropy 계산 가능


class MyDataset(Dataset):
    def __init__(self, type='train'):
        self.data, self.target = loadTraffic(type=type)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return {'data': x, 'target': y}

    def __len__(self):
        return len(self.data)

traindata = MyDataset(type = 'train')
testdata = MyDataset(type = 'test')

#data_loader = DataLoader(traindata, batch_size=1, shuffle=True)
##
class CNN_class(torch.nn.Module):

    def __init__(self, ch1, ch2, ch3, ch4):
        super(CNN_class, self).__init__()
        self.ch4 = ch4
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, ch1, kernel_size=3, stride=1, padding=1, bias=True),
            #torch.nn.BatchNorm2d(ch1),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(ch1, ch2, kernel_size=3, stride=1, padding=1, bias=True),
            #torch.nn.BatchNorm2d(ch2),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch2, ch3,kernel_size=3, stride=1, padding=1, bias=True),
            #torch.nn.BatchNorm2d(ch3),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(ch3, ch4, kernel_size=3, stride=1, padding=1, bias=True),
            #torch.nn.BatchNorm2d(ch4),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU()
        )
        self.fc_class = torch.nn.Sequential(
            torch.nn.Linear(ch4, 400, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU()
        )
        self.fc_boundingBox = torch.nn.Sequential(
            torch.nn.Linear(ch4, 400, bias=True),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU()
        )
        self.out_class = torch.nn.Linear(400,58,bias=True) # class는 58개, 원핫인코딩
        # regression 출력 4개 left_top_x, left_top_y, right_bottom_x, right_bottom_y
        self.out_boundingBox =torch.nn.Linear(400,4,bias=True)
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        #out4 = out3.view(self.ch4,-1).mean(dim=1)  # 1개 채널이 1개 fcn으로 1:1 mean pooling되어 매칭됨
        out4 = out3.view(self.ch4, -1).max(1)[0]
        out5_class, out5_boundingBox = self.fc_class(out4), self.fc_boundingBox(out4)
        class_out, bb_out = self.out_class(out5_class), self.out_boundingBox(out5_boundingBox)
        return class_out, bb_out

def calc_iou(y_bb, pred_bb):
    y_bb_area = (y_bb[2] - y_bb[0]) * (y_bb[3] - y_bb[1])
    pred_bb_area = (pred_bb[2] - pred_bb[0]) * (pred_bb[3] - pred_bb[1])
    # overlap x,y 길이 구하기
    inter_x_length = min(y_bb[2], pred_bb[2]) - max(y_bb[0], pred_bb[0])
    inter_y_length = min(y_bb[3], pred_bb[3]) - max(y_bb[1], pred_bb[1])
    # 구한 면적으로 정확도 구하기
    # 똑같으면 맞춘 것(1), 다르면 그에 맞는 비율 출력
    if ((inter_x_length > 0) & (inter_y_length > 0)):
        inter_area = inter_x_length * inter_y_length
        union_area = y_bb_area + pred_bb_area - inter_area
        iou = (inter_area / union_area).cpu().detach().numpy()
    else:
        iou = 0

    return iou
##
model = CNN_class(28,56,80,108).to(device)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)
# criterion을 통해 크로스엔트로피를 계산하는데, Softmax가 자동으로 계산되므로 fc layer에서 softmax를 쓰면 안된다
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## train
accuMatrix = np.zeros(max_epoch)
accuBBMatrix = np.zeros(max_epoch)
freq = int(len(traindata) / 100) # accumatrix_all 위한 index
accuMatrix_all = np.zeros(820)
accuMatrixbb_all = np.zeros(820)
accu_test = np.zeros(820)
accu_bb_test = np.zeros(820)
for epoch in range(max_epoch):

    pp = 0
    accu, accu_bb = 0, 0
    # data shuffle
    shfl = torch.arange(len(traindata))
    shfl = torch.randperm(max(shfl))
    tempd, templ = {},{}
    for i in range(len(shfl)) :
        tempd[i] = traindata.data[shfl.numpy()[i]]
        templ[i] = traindata.target[shfl.numpy()[i]]
    traindata.data = tempd
    traindata.target = templ

    for it in range(len(traindata)):

        #model.train()
        #classification network 학습
        x = traindata.data[it].to(device)
        x = x.unsqueeze(1).permute(1,3,0,2) # batch, channel, w, h순으로 변경
        y = traindata.target[it].to(device)
        y_class = y[6].unsqueeze(0) #class 만 추출
        y_bb = y[2:6] # bounding box 추출 left_top_x, left_bottom_y, right_bottom_x, right_top_y

        # 학습 시작
        optimizer.zero_grad()
        pred_class, pred_bb = model(x)
        cost_class = criterion(pred_class.unsqueeze(0), y_class)
        cost_bb =  F.l1_loss(pred_bb, y_bb, reduction="none").to(device)
        cost_bb = cost_bb.sum()
        cost = cost_class + cost_bb/100
        cost.backward()
        optimizer.step()
        #정확도 계산
        if pred_class.argmax() == y_class : accu += 1 # class 정확도
        accu_bb += calc_iou(y_bb, pred_bb)
        if (it % 100 == 0) & (it != 0):
            print('iter:',it, 'hit_class:',accu/100, 'hit_iou:', accu_bb/100)
            accuMatrix_all[epoch*freq + pp] = accu/100
            accuMatrixbb_all[epoch*freq + pp] = accu_bb/100

            accu, accu_bb = 0, 0
            # 테스트 정확도 계산
            with torch.no_grad():
                #model.eval()
                shfl_test = torch.arange(len(testdata))
                shfl_test = torch.randperm(max(shfl_test))
                temptd, temptl = {}, {}
                for i in range(len(shfl_test)):
                    temptd[i] = testdata.data[shfl_test.numpy()[i]]
                    temptl[i] = testdata.target[shfl_test.numpy()[i]]
                testdata.data = temptd
                testdata.target = temptl

                for it2 in range(int(len(testdata)/10)):
                    # classification network 학습
                    x = testdata.data[it2].to(device)
                    x = x.unsqueeze(1).permute(1, 3, 0, 2)  # batch, channel, w, h순으로 변경
                    y = testdata.target[it2].to(device)
                    y_class = y[6].unsqueeze(0)  # class 만 추출
                    y_bb = y[2:6]  # bounding box 추출 left_top_x, left_bottom_y, right_bottom_x, right_top_y

                    pred_class, pred_bb = model(x)
                    if pred_class.argmax() == y_class:
                        accu += 1  # class 정확도
                    accu_bb += calc_iou(y_bb, pred_bb)
                print('epoch:', epoch, 'accutest:', accu / it2, 'accutest_iou:', accu_bb / it2)
                accu_test[epoch*freq + pp] = accu/it2
                accu_bb_test[epoch*freq + pp] = accu_bb/it2
                accu, accu_bb = 0, 0
                pp += 1


##
m = 0
m1 ,m2 ,m3, m4 = np.zeros(41), np.zeros(41), np.zeros(41),np.zeros(41)
for i in range(0,len(accuMatrixbb_all),20):
    m1[m] = accuMatrix_all[i:i+20].mean()
    m2[m] = accuMatrixbb_all[i:i+20].mean()
    m3[m] = accu_test[i:i+20].mean()
    m4[m] = accu_bb_test[i:i+20].mean()
    m+=1

##
g = np.linspace(0, 20, len(accuMatrixbb_all))
g2 = np.linspace(0, 20, 41)
plt.scatter(g,accuMatrix_all[0:820],s=0.5, c='red')
plt.scatter(g,accuMatrixbb_all[0:820],s=0.5, c='magenta')
plt.scatter(g,accu_test[0:820],s=0.5, c='blue')
plt.scatter(g,accu_bb_test[0:820],s=0.5, c='skyblue')

plt.plot(g2, m1, c='red', lw=3)
plt.plot(g2, m2, c='magenta', lw=3)
plt.plot(g2, m3, c='blue', lw=3)
plt.plot(g2, m4, c='skyblue', lw=3)

plt.legend(["accu_class", "accu_BB", "test_accu_class", "test_accu_BB"])
plt.xlabel('epoch')
plt.grid(1)
plt.show()

##

