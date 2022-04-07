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
        self.in_embeddings = torch.nn.Embedding(vocabulary_size,embedding_dim)#返回（batch,num，feature_dim）
        self.out_embeddings = torch.nn.Embedding(vocabulary_size,embedding_dim)
        initrange = 0.5/embedding_dim# 单个词向量权值之和在-0.5和0.5之间
        self.in_embeddings.weight.data.uniform_(-initrange,initrange)# 权值按均匀分布初始化
        self.out_embeddings.weight.data.uniform_(-initrange,initrange)
        self.win_size = win_size
    
    def forward(self,centers,context,neg_context):
        # 这里的centers跟context就是中心词和背景词，而context的背景词
        # 输入是按一个中心词一个背景词输入的
        # 这里使用了两个embedding矩阵，他把中心词和背景词都乘上这个矩阵，变成压缩向量以后
        # ，内积得到的结果计算loss
        # 文献3中只用了一个embedding层，也就是说它认为一个词作为中心词和作为背景词的映射方式是一样的
        batch_size = len(centers)#[batch_size,feature_dim]
        u_embeds = self.in_embeddings(centers).view(batch_size,1,-1)#为了方便做乘法[batch,1,dim]
        v_embeds = self.out_embeddings(context).view(batch_size,2*self.win_size,-1)#[batch,2*self.win_size,dim]
        score = torch.bmm(u_embeds,v_embeds.transpose(1,2)).squeeze()#矩阵相乘
        pos_score = torch.mean(score,dim=1)#batch,2*self.win_size
        loss = F.logsigmoid(pos_score).squeeze()
        #print(loss.shape)
        neg_v_embeds = self.out_embeddings(neg_context)
        neg_score = torch.bmm(u_embeds,neg_v_embeds.transpose(1,2)).squeeze()
        neg_score = torch.mean(neg_score,dim=1)# 一个中心词有多个背景词，求和
        neg_score = F.logsigmoid(-1*neg_score).squeeze()
        loss+=neg_score
        #print(loss.shape)
        return -1*loss.mean()
    def get_embeddings(self):
        return self.in_embeddings.weight.data

if __name__=='__main__':
    from SkipGramDataset import WordEmbeddingDataset,get_word_freq
    # 超参数
    BATCH_SIZE=2
    EPOCHS=1000
    # 文件读取
    with open('text8.txt') as f:
        words = f.read()
    words=words[:100000]
    print('len of words=',len(words))
    words = words.split(" ")
    id_word,word_freq = get_word_freq(words,VOCABULARY_SIZE=5000)
    print('len of vocabulary:',len(word_freq))

    # 创建数据集
    print('can use cuda:',torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wed = WordEmbeddingDataset(data=id_word,word_freq=word_freq)
    dataloader = tud.DataLoader(wed,batch_size=64,shuffle=True)
    model = SkipGramModel(vocabulary_size=len(word_freq),embedding_dim=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 训练
    for epoch in range(EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            input_labels = input_labels.long().to(device)  # 全部转为LongTensor
            pos_labels = pos_labels.long().to(device)
            neg_labels = neg_labels.long().to(device)
            optimizer.zero_grad()  # 梯度归零
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print("epoch", epoch, "loss", loss.item())
'''
代码输出
epoch 0 loss 1.3862918615341187
epoch 50 loss 1.3812416791915894
epoch 100 loss 1.2875112295150757
epoch 150 loss 1.2195510864257812
epoch 200 loss 1.0609501600265503
epoch 250 loss 1.2624115943908691
epoch 300 loss 1.070410966873169
epoch 350 loss 0.9761687517166138
epoch 400 loss 0.9282042384147644
epoch 450 loss 0.8098894953727722
epoch 500 loss 0.6932863593101501
epoch 550 loss 0.6844012141227722
epoch 600 loss 0.7273381948471069
epoch 650 loss 0.8231338262557983
epoch 700 loss 0.585033655166626
epoch 750 loss 0.4203912317752838
epoch 
'''