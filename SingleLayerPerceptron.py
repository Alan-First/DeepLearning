'''
Description: Build a Single-Layer Perceptron
Version: 2.0
Autor: Hedwig
Date: 2022-03-29 15:59:11
LastEditors: Hedwig
LastEditTime: 2022-03-29 20:15:06
'''
import numpy as np
import matplotlib.pyplot as plt
import math

#备注：这是一个练手的代码，难度不大基本diy，测试用例和可视化来自覃秉丰的《深度学习从0到1》
#      主要就是实现单层感知机的训练过程和测试并可视化
def sigmoid(Y):
    return 1/(1+math.exp(Y))

def sign(Y):
    Y[Y>=0]=1
    Y[Y<0]=-1
    return Y

class SingleLayerPerceptron(object):
    def __init__(self,feature_dim):
        self.w = np.random.random([feature_dim+1,1])#维度扩展实现wx+b
    
    def activate(self,X):
        return sign(X)

    def __call__(self,X):
        return self.activate(np.dot(X,self.w))#[num,feature_dim+1]*[feature_dim+1,1]

def train(X,Y,net,lr,epoch):
    X = np.column_stack((np.ones((X.shape[0],1)),X))#维度扩展，添加用矩阵的方式实现wx+b，不扩展则只有w*x
    for i in range(epoch):
        pred = net(X)
        if(Y==pred).all():
            print("Finished")
            break
        error = Y-pred#[num,1]
        delta_w=lr*X.T.dot(error)/X.shape[0]#[feature_dim+1,num]*[num,1]样本数量维湮没，压缩成一个数
        net.w = net.w+delta_w
        print('epoch',i+1)
        print('weight',net.w)
    return net.w,net(X)

def visual(X,w):
    k=-w[1]/w[2]
    b=-w[0]/w[2]#把二维的w转化成y=kx+b的直线
    sample = (0,5)#画直线用的点
    plt.plot(sample,sample*k+b,'r')
    x_p = [X[0][0],X[1][0]]
    y_p = [X[0][1],X[1][1]]
    x_n = [X[2][0],X[3][0]]
    y_n = [X[2][1],X[3][1]]

    plt.scatter(x_p,y_p,c='b')
    plt.scatter(x_n,y_n,c='y')
    plt.show()

if __name__ == '__main__':
    X=np.array([[3,3],[4,3],[1,1],[2,1]])#数据shape为[num,feature_dim]
    Y = np.array([[1],[1],[-1],[-1]])#数据shape为[num,1]
    SLP = SingleLayerPerceptron(feature_dim=X.shape[1])
    w,y = train(X,Y,net=SLP,lr=0.1,epoch=100)
    print('y=',y)
    visual(X,w)

