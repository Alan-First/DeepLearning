#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :SkipGram.py
@Description  :代码基础来自张楠的《深度学习自然语言处理实战》，原代码功能不完善，有修改
@Time         :2022/04/06 11:42:51
@Author       :Hedwig
@Version      :1.0
'''
import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.data as tud
import collections


# 搭建跳字模型，从中心词推断背景词
# 代码来自文献3，改动了其中负采样表的实现
class SkipGramModel(torch.nn.Module):
    def __init__(self,vocabulary_size,embedding_dim,win_size=1):
        super(SkipGramModel,self).__init__()
        self.embeddings = torch.nn.Embedding(vocabulary_size,embedding_dim)#返回（batch,num，feature_dim）
        initrange = 0.5/embedding_dim# 单个词向量权值之和在-0.5和0.5之间
        self.embeddings.weight.data.uniform_(-initrange,initrange)# 权值按均匀分布初始化
        self.win_size = win_size
    
    def forward(self,centers,context,neg_context):
        # 这里的centers跟context就是中心词和背景词，而context的背景词
        # 输入是按一个中心词一个背景词输入的
        # 这里只使用了一个embedding矩阵，他把中心词和背景词都乘上这个矩阵，变成压缩向量以后
        # ，内积得到的结果计算loss
        batch_size = len(centers)#[batch_size,feature_dim]
        u_embeds = self.embeddings(centers).view(batch_size,1,-1)#为了方便做乘法[batch,1,dim]
        v_embeds = self.embeddings(context).view(batch_size,2*self.win_size,-1)#[batch,2*self.win_size,dim]
        score = torch.bmm(u_embeds,v_embeds.transpose(1,2)).squeeze()#矩阵相乘
        pos_score = torch.sum(score,dim=1)#batch,1,2*self.win_size
        loss = F.logsigmoid(pos_score).squeeze()
        neg_v_embeds = self.embeddings(neg_context)
        neg_score = torch.bmm(u_embeds,neg_v_embeds.transpose(1,2)).squeeze()
        neg_score = torch.sum(neg_score,dim=1)# 一个中心词有多个背景词，求和
        neg_score = F.logsigmoid(-1*neg_score).squeeze()
        loss+=neg_score
        return -1*loss.sum()
    def get_embeddings(self):
        return self.embeddings.weight.data

if __name__=='__main__':
    from SkipGramDataset import WordEmbeddingDataset
    # 超参数
    BATCH_SIZE=2
    EPOCHS=100
    dataloader = tud.DataLoader(batch_size=2,shuffle=True)
    model = SkipGramModel(vocabulary_size=len(word_freq),embedding_dim=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for i in range(EPOCHS)