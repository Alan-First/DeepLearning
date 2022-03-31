'''
Description: pytorch中张量计算图的实现，基本来自张伟振的《深度学习原理与Pytorch实战》，\\
    原文中代码不够完整有报错，小修了一下使其可以运行
Version: 
Autor: Hedwig
Date: 2022-03-31 16:25:56
LastEditors: Hedwig
LastEditTime: 2022-03-31 20:46:33
'''
from Graph import Node,Graph
from abc import ABCMeta, abstractmethod
graph = Graph()

# 创建一个继承节点类的运算类作为后续操作的父类
# 类本身想要前向和后向操作
class Function(Node):
    def forward(self,*argv):
        pass
    
    @abstractmethod
    def backward(self,grad):
        pass

# 创建张量类，张量类本身记录自身数据大小和梯度，同时还有前后向的节点信息
class Tensor(Function):
    def __init__(self,data):
        super().__init__()#IndexGetter.get_id()
        self.data = data
        self.grad = 0
    
    # 张量本身是一个数，不是函数，没有前向的说法
    # 它的反向
    def backward(self,grad=1):
        self.grad += grad
        for previous_node in self.previous_nodes:
            previous_node.backward(grad)


# 乘法类，前向就是两数相乘
class Multiply(Function):
    def forward(self,x:Tensor,y:Tensor)->Tensor:
        result_tensor = Tensor(x.data * y.data)# 先做一次乘法
        graph.add_edge(self,result_tensor)# 乘法操作与乘法结果为新的节点加入图中
        return result_tensor

    # 反向就是求导，x*y求导，对x求导是y，对y求导是x
    def backward(self,grad):
        x_node:Tensor = self.previous_nodes[0]# 乘法的前驱节点0位置是x，因为multiply先添加的x
        y_node:Tensor = self.previous_nodes[1]
        x_node.backward(grad * y_node.data)# 由于乘法的链式求导法则，反向时候是累乘的
        y_node.backward(grad * x_node.data)

def multiply(x:Tensor,y:Tensor):
    multiply_node = Multiply()#IndexGetter.get_id()
    graph.add_edge(x,multiply_node)# 先添加的x
    graph.add_edge(y,multiply_node)# 后添加的y
    return multiply_node.forward(x,y)

class Add(Function):
    def forward(self,x:Tensor,y:Tensor)->Tensor:
        result_tensor = Tensor(x.data+y.data)
        graph.add_edge(self,result_tensor)
        return result_tensor

    # 加法的导数是1
    def backward(self, grad):
        x_node:Tensor = self.previous_nodes[0]
        y_node:Tensor = self.previous_nodes[1]
        x_node.backward(grad*1)
        y_node.backward(grad*1)

def add(x:Tensor, y:Tensor):
    add_node = Add()#IndexGetter.get_id()
    #print(add_node.id)
    graph.add_edge(x,add_node)
    graph.add_edge(y,add_node)
    return add_node.forward(x,y)

if __name__=='__main__':
    w=Tensor(2)
    x=Tensor(1)
    b=Tensor(0.5)
    y=add(multiply(w,x),b)
    y.backward()
    print('w.grad:{}'.format(w.grad))
    print('x.grad:{}'.format(x.grad))
    print('y.grad:{}'.format(y.grad))