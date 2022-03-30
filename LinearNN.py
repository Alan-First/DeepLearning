'''
Description: Build a linear neural network
Version: 2.0
Autor: Hedwig
Date: 2022-03-29 20:16:55
LastEditors: Hedwig
LastEditTime: 2022-03-29 20:28:52
'''
import numpy as np
import matplotlib.pyplot as plt
import math
from SingleLayerPerceptron import SingleLayerPerceptron

# 这里线性神经网络模型其实就是单层感知机的输出层取消激活函数的结果
# main代码是为了直观说明：网络的非线性可以解决异或问题
# 代码中体现了：线性神经网络不能解决的异或问题，在将特征非线性组合后可以被解决
# 由此可知：如果网络是非线性的，具有把特征整合成非线性的功能，也是可以解决这个问题的
class LinearNN(SingleLayerPerceptron):
    """
    线性神经网络模型与单层感知机区别只在于激活函数是纯线性函数，所以这里我继承了单层感知机重写激活函数
    """
    def activate(self,X):
        return X

def train(X,Y,net,lr,epoch=100):
    X = np.column_stack((np.ones((X.shape[0],1)),X))#维度扩展，添加用矩阵的方式实现wx+b，不扩展则只有w*x
    for _ in range(epoch):
        pred = net(X)
        error = Y-pred#[num,1]
        print(error)
        delta_w=lr*X.T.dot(error)/X.shape[0]#[feature_dim+1,num]*[num,1]样本数量维湮没，压缩成一个数
        net.w = net.w+delta_w
    return net.w,net(X)
'''
# 这部分代码跟单层感知机一样，运行结果也类似
# 很明显，这种生成线性的分类无法处理异或问题，因为最终结果是一条直线
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
    LNN = LinearNN(feature_dim=X.shape[1])
    w,y = train(X,Y,net=LNN,lr=0.1,epoch=100)
    print('y=',y)
    visual(X,w)
'''
def root(w,x,root):
    a=w[5]
    b=w[2]+x*w[4]
    c=w[0]+x*w[1]+x*x*w[3]
    if root == 1:
        return (-b+np.sqrt(b**2-4*a*c))/(2*a)
    if root == 2:
        return (-b-np.sqrt(b**2-4*a*c))/(2*a) 


def visual(w):
    # 先看下面的main函数再看这个可视化函数
    # main函数我们把样本特征映射到了非线性的空间，在这里我们把它映射回线性空间
    # 在线性空间里，样本依然是二维特征的样本
    # 在非线性空间求出来的直线解析式是：w0+w1x+w2y+w3x^2+w4xy+w5y^2=0
    # 在已知我们是把(x,y)->(x,y,x^2,xy,y^2)的基础上
    # 函数为w5 * y^2 + (w2+w4*x) * y + (w0+w1x+w3x^2) = 0
    # 把它看成y关于x的函数，用二次函数求根公式可以解出y与x的关系
    # 它是一条双曲线
    x_p = [0,1]
    y_p = [1,0]#正样本的横纵坐标都不一样
    x_n = [0,1]
    y_n = [0,1]#负样本的横纵坐标都一样
    xnum = np.linspace(-0.5,1.5,100)
    plt.plot(xnum,root(w,xnum,1),'r')
    plt.plot(xnum,root(w,xnum,2),'r')
    plt.scatter(x_p,y_p,c='b')
    plt.scatter(x_n,y_n,c='y')
    plt.show()

# 对于测试用例，特征(x0,x1)分别为(0,0),(0,1),(1,0),(1,1)，异或问题下的标签分别为-1，1，1，-1
# 线性神经网络无法解决这类问题，通过把特征做非线性组合是一种解决思路
# 比如将特征拓展为(x0,x1,x0^2,x0*x1,x1^2)，这时候特征就变成了
# (0,0,0,0,0),(0,1,0,0,1),(1,0,1,0,0),(1,1,1,1,1)
# 测试用例如下
if __name__ == '__main__':
    X=np.array([[0,0,0,0,0],[0,1,0,0,1],[1,0,1,0,0],[1,1,1,1,1]])#数据shape为[num,feature_dim]
    Y = np.array([[-1],[1],[1],[-1]])#数据shape为[num,1]
    LNN = LinearNN(feature_dim=X.shape[1])
    w,y = train(X,Y,net=LNN,lr=0.1,epoch=100)
    print('y=',y)
    visual(w)
