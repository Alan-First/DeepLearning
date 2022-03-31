#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :MultipleLayerPerceptron.py
@Description  :实现多层感知机及其反向传播，主要是DIY，测试用例和可视化来自于覃秉丰的《深度学习从0到1》
@Time         :2022/03/30 10:13:16
@Author       :Hedwig
@Version      :1.0
'''
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1/(1+np.exp(-X))

def dsigmoid(X):
    return X*(1-X)

# 多层感知机层数是指神经元层数，每一层都有一个输出
# 比如三层感知机有两个隐藏层和一个输出层；双层感知机是一个隐藏层和一个输出层
# 单层感知机隐藏层就是输出层
# 这个感知机中间层维度都是统一的，而且输出只有一维
class MultipleLayerPerceptron:
    def __init__(self,feature_dim,layer_num=2,hide_dim=10):
        # 有多少层就有多少个待确定的矩阵
        if layer_num==1:
            self.MLP = [np.random.random([feature_dim,1])]
            self.b = [np.random.random([1])]
        elif layer_num>=2:
            self.MLP = [np.random.random([feature_dim,hide_dim])]
            self.b = [np.random.random([hide_dim]) for _ in range(layer_num-1)]
            for _ in range(layer_num-2):
                self.MLP.append(np.random.random([hide_dim,hide_dim]))  
            self.MLP.append(np.random.random([hide_dim,1]))
            self.b.append(np.random.random([1]))
        else:
            raise('Value of layer_num should be not less than 1')
        
    def activate(self,X):
        return sigmoid(X)

    def __call__(self,X):
        Y=[]
        for idx,layer in enumerate(self.MLP):
            #print(X.shape)
            #print(layer.shape)
            X = self.activate(np.dot(X,layer)+self.b[idx])
            Y.append(X)# Y的尺寸跟感知机层数一样
        return Y

def update(net,lr,pred,X,Y):
    '''
    多层感知机的第h层的变化量是：学习率*上一层输出矩阵转置*本层学习信号
    对于隐藏层，本层学习信号=下一层学习信号*下一层权值矩阵.*本层输出的导数
    对于输出层，本层学习信号=输出信号与标签的残差.*本层输出的导数
    '''
    sign = [0]*(len(pred))#sign记录了每个隐藏层和输出层的信号
    sign[-1] = (Y-pred[-1])*dsigmoid(pred[-1])#输出层的信号量
    for i in range(1,len(pred)):
        sign[-i-1]=sign[-i].dot(net.MLP[-i].T)*dsigmoid(pred[-i-1])#各隐藏层的信号量
    
    delta_w = [0]*(len(pred))#记录各个隐藏层和输出层权值变化量
    for i in range(len(pred)-1):
        delta_w[-i-1]=lr*pred[-i-2].T.dot(sign[-i-1])/X.shape[0]
    delta_w[0]=lr*X.T.dot(sign[0])/X.shape[0]

    for i in range(len(pred)):#将各个隐藏层和输出层权值更新
        net.MLP[i] += delta_w[i]
        net.b[i] += lr*np.mean(sign[i],axis=0)

def train(X,Y,net,lr,epoch):
    loss=[]
    for i in range(epoch):
        pred = net(X)
        if i%2500==0:
            print('epoch:',i+1,'loss:',np.mean(np.square(Y-pred[-1])/2),'res',pred[-1])
            loss.append(np.mean(np.square(Y-pred[-1])/2))
        if(Y==pred[-1]).all():
            print("Finished")
            break
        update(net,lr,pred,X,Y)
        
    return loss
        
def visual(loss,epoch):
    plt.plot(range(0,epoch,2500),loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    X=np.array([[0,0],[0,1],[1,0],[1,1]])#数据shape为[num,feature_dim]
    Y = np.array([[0],[1],[1],[0]])#数据shape为[num,1]
    # 当多层感知机为2层时，隐藏层神经元为10维，运行100000次，loss约为0.00027/0.01422/0.00026/0.00023
    # 当多层感知机为3层时，隐藏层神经元为10维，运行100000次，loss约为0.00044/0.00048/0.00062/0.00031
    # 当多层感知机为2层，隐藏层神经元为20维，运行100000次，loss约为0.00027/0.00023/0.00025/0.00028
    # 性能都差不多 应该是参数量足够了
    MLP = MultipleLayerPerceptron(feature_dim=X.shape[1],layer_num=2,hide_dim=5)
    epoch=100000
    loss = train(X,Y,net=MLP,lr=0.1,epoch=epoch)
    visual(loss,epoch)