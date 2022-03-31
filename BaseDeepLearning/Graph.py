'''
Description: 基代码来自张伟振的《深度学习原理与Pytorch实战》，有修改，原始代码修改部分用绿色#注释掉
Version: 
Autor: Hedwig
Date: 2022-03-31 16:11:17
LastEditors: Hedwig
LastEditTime: 2022-03-31 22:06:29
'''
# 用邻接表实现图
# Node类对象记录每个节点，用列表记录节点的上一个节点和下一个节点
# 本实现来自于张伟振的《深度学习原理与Pytorch实战》
# 这部分主要是模拟pytorch张量的计算
class Node:
    def __init__(self):
        #self.id = id(self)# 这个属性值作为区分对象的标志
        self.previous_nodes = [] # 前驱节点
        self.next_nodes = [] #后续节点

# 先创建Node对象和Graph对象，用add_edge方法实现对左右节点的链接，这种链接时按键值对实现的
# 字典的键是id，值是节点对象
class Graph:
    def __init__(self):
        #self.Nodes = {}
        self.Nodes = set()
    def add_edge(self,head:Node,tail:Node)->Node:
        """
        @Params :要链接的两个节点head和tail
        
        @Description :将head指向tail，并将head和tail都存进有向图字典中，id:节点按键值对存储
        
        @Returns     :
        """
        # 这部分是书上的代码，被修改
        #if head.id not in self.Nodes:
        #    self.Nodes[head.id] = head
        #if tail.id not in self.Nodes:
        #    self.Nodes[tail.id] = tail
        #self.Nodes[head.id].next_nodes.append(tail)
        #self.Nodes[tail.id].previous_nodes.append(head)
        head.next_nodes.append(tail)
        tail.previous_nodes.append(head)
        self.Nodes.add(head)
        self.Nodes.add(tail)

    #def traverse(self):
    #    for node_id in self.Nodes:
    #        print("{}:{}".format(node_id,','.join([str(next_node.id) \
    #            for next_node in self.Nodes[node_id].next_nodes])))






