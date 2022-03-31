'''
Description: 
Version: 
Autor: Hedwig
Date: 2022-03-31 16:11:17
LastEditors: Hedwig
LastEditTime: 2022-03-31 19:02:40
'''

# 用邻接表实现图
# Node类对象记录每个节点，用列表记录节点的上一个节点和下一个节点
# 本实现来自于张伟振的《深度学习原理与Pytorch实战》
# 这部分主要是模拟pytorch张量的计算
class Node:
    def __init__(self,idx=0):
        self.id = id(self)
        self.previous_nodes = []
        self.next_nodes = []

# 先创建Node对象和Graph对象，用add_edge方法实现对左右节点的链接，这种链接时按键值对实现的
# 字典的键是id，值是节点对象
class Graph:
    def __init__(self):
        self.Nodes = {}
    def add_edge(self,tail:Node,head:Node)->Node:
        if tail.id not in self.Nodes:
            self.Nodes[tail.id] = tail
        if head.id not in self.Nodes:
            self.Nodes[head.id] = head
        self.Nodes[tail.id].next_nodes.append(head)
        self.Nodes[head.id].previous_nodes.append(tail)
    def traverse(self):
        for node_id in self.Nodes:
            print("{}:{}".format(node_id,','.join([str(next_node.id) \
                for next_node in self.Nodes[node_id].next_nodes])))






