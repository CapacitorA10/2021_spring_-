## Q-1
import numpy as np
A = np.random.random((5,5))
b = np.random.random((5,1))

detA = np.linalg.det(A)
print("행렬식: ",detA)

eigenvalA, eigenvecA = np.linalg.eig(A)
print("\n고유값: ", eigenvalA)
print("\n고유벡터: ",eigenvecA)

x = b.transpose() @ np.linalg.inv(A)
x = x.transpose()
print("\nAx=b에서의 x행렬:",x)

## Q-2
import matplotlib.pyplot as plt
L, N = 100, 1000
mu1, sigmaSquare1 = -30, 200
mu2, sigmaSquare2 = 20, 400
gaus1 = np.random.normal(mu1, np.sqrt(sigmaSquare1), N)
gaus2 = np.random.normal(mu2, np.sqrt(sigmaSquare2), N)

# 엔트로피 계산
hist1 = np.zeros((2*L,1))
hist2 = np.zeros((2*L,1))
for i in range(N):
    k = gaus1[i]
    k = int(k.round()) + L
    hist1[k] = hist1[k] + 1

    j = gaus2[i]
    j = int(j.round()) + L
    hist2[j] = hist2[j] + 1

axisX = np.linspace(-L,L,2*L)
plt.bar(axisX, hist1.squeeze(), label="gaussian 1")
plt.bar(axisX, hist2.squeeze(), label="gaussian 2")
plt.legend(loc='upper right')

## Q-2-1
etrp1 = 0
etrp2 = 0
for i in range(2*L):
    p1 = hist1[i]/ N
    etrp1 += -( p1 * np.log2(p1+0.00001) )
    p2 = hist2[i] / N
    etrp2 += -( p2 * np.log2(p2+0.00001) )

print('\n1번 엔트로피 : ',etrp1)
print('\n2번 엔트로피 : ',etrp2)

## Q-2-2

crossEntropy = 0
KLdv = 0
for i in range(2*L):
    px = hist1[i]/ N
    qx = hist2[i]/ N
    crossEntropy += -( px * np.log2(qx+0.00001) )

print("\n Crossentropy : ", crossEntropy)

for i in range(2*L):
    px = (hist1[i]/ N)
    qx = (hist2[i]/ N)
    KLdv += -( px * np.log2( (px/(qx+0.00001))+0.00001 ) )

print("\n KL다이버전스 : ", KLdv)

## Q-3-1

q3_x = np.zeros([3,1000]) # x = [x1,x2,x3]가 1000개
q3_x[:,0] -= 1
rho = 0.1

for i in range(len(q3_x[0,]) - 1):
    # x - rho(x편미분) 를 x1, x2, x3에 대해 적용
    q3_x[0, i+1] = q3_x[0, i] - rho * (2*q3_x[0, i] - 2)
    q3_x[1, i+1] = q3_x[1, i] - rho * 4*(2*q3_x[1, i] - 4)
    q3_x[2, i+1] = q3_x[2, i] - rho * 9*(2*q3_x[2, i] - 6)

plt.subplot(121)
x_range = np.arange(1,1001)
plt.plot(x_range, q3_x[0], 'r', label='x_1')
plt.plot(x_range, q3_x[1], 'g', label='x_2')
plt.plot(x_range, q3_x[2], 'b', label='x_3')
plt.xlabel('iteration')
plt.ylabel('value')
plt.legend(loc='upper right')


plt.subplot(122)
x_range2 = np.arange(1,31)
plt.plot(x_range2, q3_x[0,:30], 'r', label='x_1')
plt.plot(x_range2, q3_x[1,:30], 'g', label='x_2')
plt.plot(x_range2, q3_x[2,:30], 'b', label='x_3')
plt.xlabel('iteration')
plt.ylabel('value')
plt.legend(loc='upper right')
plt.show()

##q4
q3_x = np.zeros([3,1000]) # x = [x1,x2,x3]가 1000개
q3_x[:,0] -= 10
rho = 0.002
for i in range(len(q3_x[0,]) - 1):
    # x - rho(x편미분) 를 x1, x2, x3에 대해 적용
    q3_x[0, i+1] = q3_x[0, i] - rho * (2*q3_x[0, i] - 2)
    q3_x[1, i+1] = q3_x[1, i] - rho * 4*(2*q3_x[1, i] - 4)
    q3_x[2, i+1] = q3_x[2, i] - rho * 9*(2*q3_x[2, i] - 6)
x_range = np.arange(1,1001)
plt.subplot(211)
plt.plot(x_range, q3_x[0], 'r', label='x_1')
plt.plot(x_range, q3_x[1], 'g', label='x_2')
plt.plot(x_range, q3_x[2], 'b', label='x_3')
plt.xlabel('iteration'); plt.ylabel('value'); plt.legend(loc='upper right')


q3_x = np.zeros([3,1000]) # x = [x1,x2,x3]가 1000개
q3_x[:,0] -= 1000
rho = 0.002
for i in range(len(q3_x[0,]) - 1):
    # x - rho(x편미분) 를 x1, x2, x3에 대해 적용
    q3_x[0, i+1] = q3_x[0, i] - rho * (2*q3_x[0, i] - 2)
    q3_x[1, i+1] = q3_x[1, i] - rho * 4*(2*q3_x[1, i] - 4)
    q3_x[2, i+1] = q3_x[2, i] - rho * 9*(2*q3_x[2, i] - 6)


x_range = np.arange(1,1001)
plt.subplot(212)
plt.plot(x_range, q3_x[0], 'r', label='x_1')
plt.plot(x_range, q3_x[1], 'g', label='x_2')
plt.plot(x_range, q3_x[2], 'b', label='x_3')
plt.xlabel('iteration'); plt.ylabel('value'); plt.legend(loc='upper right')
plt.show()


##


##

