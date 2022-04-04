#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :SkipGram.py
@Description  :搭建带有负采样的跳字模型，代码主要来自张楠的《深度学习自然语言处理实战》
@Time         :2022/04/02 21:32:45
@Author       :Hedwig
@Version      :1.0
'''
from turtle import forward
import torch
import torch.nn.functional as F
import numpy as np

TABLE_SIZE = 1e8

def create_sample_table(word_count):
    """
    @Params :
    
    @Description :创建负采样表
    
    @Returns     :
    """
    table = []
    frequency = np.power(np.array(word_count),0.75)
    sum_frequency = sum(frequency)
    ratio = frequency/sum_frequency
    count = np.round(ratio*TABLE_SIZE)
    for word_idx,c in enumerate(count):
        table+=[word_idx]*int(c)
    return np.array(table)

# 搭建跳字模型，从中心词推断背景词
class SkipGramModel(torch.nn.Module):
    def __init__(self,device,vocabulary_size,embedding_dim,neg_num=0,word_count=[]):
        super(SkipGramModel,self).__init__()
        self.device = device
        self.neg_num = neg_num
        self.embeddings = torch.nn.Embedding(vocabulary_size,embedding_dim)#返回（batch,num，feature_dim）
        initrange = 0.5/embedding_dim# 单个词向量权值之和在-0.5和0.5之间
        self.embeddings.weight.data.uniform_(-initrange,initrange)# 权值按均匀分布初始化
        if self.neg_num>0:
            self.table = create_sample_table(word_count)
    
    def forward(self,centers,context):
        batch_size = len(centers)#[batch_size,feature_dim]
        u_embeds = self.embeddings(centers).view(batch_size,1,-1)#为了方便做乘法
        v_embeds = self.embeddings(context).view(batch_size,1,-1)
        score = torch.bmm(u_embeds,v_embeds.transpose(1,2)).squeeze()#矩阵相乘
        loss = F.logsigmoid(score).squeeze()
        if self.neg_num>0:
            neg_contexts = torch.LongTensor(np.random.choice(self.table,size=(batch_size,self.neg_num))).to(self.device)
            neg_v_embeds = self.embeddings(neg_contexts)
            neg_score = torch.bmm(u_embeds,neg_v_embeds.transpose(1,2)).squeeze()
            neg_score = torch.sum(neg_score,dim=1)
            neg_score = F.logsigmoid(-1*neg_score).squeeze()
            loss+=neg_score
        return -1*loss.sum()
    def get_embeddings(self):
        return self.embeddings.weight.data
    






