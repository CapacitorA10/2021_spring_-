import numpy as np
import matplotlib.pyplot as mplot

## 공분산 및 평균
x = np.asarray([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
                 [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2]])
mu = np.mean(x, axis=0)

covar = np.zeros([4,4])
for i in range(len(x)):
    covar += (x[i] - mu).reshape((4,1)) @ (x[i] - mu).reshape((1,4))


## 가우시안

mean = 0
sigma = 0.5
axisX = np.linspace(-5,5, 1000)
g = np.e / np.sqrt(2 * np.pi * sigma * sigma)
a = 2 * sigma * sigma
u = -(axisX - np.asarray(mean)) * (axisX - mean)

gaussian = np.power(g, u/a)
mplot.plot(axisX, gaussian)
np.mean(gaussian * axisX)
np.var(gaussian * axisX)


imsi3 = np.zeros([200,200])
lenth = len(imsi3)
for i in range(0, lenth, 1):
    #if i == 191:
        #break
    stride = lenth - 1 - i
    if i % 2 == 0 :
        imsi3[:, i] = 255
        imsi3[i, :] = 255
        imsi3[stride, :] = 255
        imsi3[:, stride] = 255
    else :
        imsi3[:, i] = 0
        imsi3[i, :] = 0
        imsi3[stride, :] = 0
        imsi3[:, stride] = 0


mplot.imshow(imsi3)