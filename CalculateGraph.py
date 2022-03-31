'''
Description: 
Version: 
Autor: Hedwig
Date: 2022-03-31 16:25:56
LastEditors: Hedwig
LastEditTime: 2022-03-31 17:12:30
'''
from Graph import Node,Graph

graph = Graph()

# 创建一个继承节点类的运算类作为后续操作的父类
class Function(Node):
    def forward(self,*argv):
        pass

    def backward(self,grad):
        pass

class Tensor(Function):
    def __init__(self,data):
        super().__init__()#IndexGetter.get_id()
        self.data = data
        self.grad = 0
    
    def backward(self,grad=1):
        self.grad += grad
        for previous_node in self.previous_nodes:
            previous_node.backward(grad)


class Multiply(Function):
    def forward(self,x:Tensor,y:Tensor)->Tensor:
        result_tensor = Tensor(x.data * y.data)
        graph.add_edge
        return result_tensor

    def backward(self,grad):
        x_node:Tensor = self.previous_nodes[0]
        y_node:Tensor = self.previous_nodes[1]
        x_node.backward(grad * y_node.data)
        y_node.backward(grad * x_node.data)

def multiply(x:Tensor,y:Tensor):
    multiply_node = Multiply()#IndexGetter.get_id()
    graph.add_edge(x,multiply_node)
    graph.add_edge(y,multiply_node)
    return multiply_node.forward(x,y)

class Add(Function):
    def forward(self,x:Tensor,y:Tensor)->Tensor:
        result_tensor = Tensor(x.data+y.data)
        graph.add_edge(self,result_tensor)
        return result_tensor

    def backward(self, grad):
        x_node:Tensor = self.previous_nodes[0]
        y_node:Tensor = self.previous_nodes[1]
        x_node.backward(grad*1)
        y_node.backward(grad*1)

def add(x:Tensor, y:Tensor):
    add_node = Add()#IndexGetter.get_id()
    graph.add_edge(x,add_node)
    graph.add_edge(y,add_node)
    return add_node.forward(x,y)

if __name__=='__main__':
    w=Tensor(2)
    x=Tensor(1)
    b=Tensor(0.5)
    y=add(multiply(w,x)+b)
    y.backward()
    print('w.grad:{}'.format(w.grad))
    print('x.grad:{}'.format(x.grad))
    print('y.grad:{}'.format(y.grad))