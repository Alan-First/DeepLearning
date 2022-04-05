#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :SkipGram.py
@Description  :搭建带有负采样的跳字模型，代码基础来自张楠的《深度学习自然语言处理实战》，原代码功能不完善，有修改
@Time         :2022/04/02 21:32:45
@Author       :Hedwig
@Version      :1.0
'''
#from turtle import forward
import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.data as tud
import collections

# 搭建跳字模型，从中心词推断背景词
class SkipGramModel(torch.nn.Module):
    def __init__(self,device,vocabulary_size,embedding_dim,win_size=1):
        super(SkipGramModel,self).__init__()
        self.device = device
        self.embeddings = torch.nn.Embedding(vocabulary_size,embedding_dim)#返回（batch,num，feature_dim）
        initrange = 0.5/embedding_dim# 单个词向量权值之和在-0.5和0.5之间
        self.embeddings.weight.data.uniform_(-initrange,initrange)# 权值按均匀分布初始化
        self.win_size = win_size
        #self.table = create_sample_table(word_count)
    
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
    

# 很多代码实现都没有考虑负采样的时候，负样本不能是正样本这种情况，而是直接从词典里按频率分布取了数
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,data,win_size=2,neg_ratio=2,word_freq=[]):
        """
        @Params :data是输入的句子（已转id），word_freq是词典的词频
        
        @Description :
        
        @Returns     :
        """
        super(WordEmbeddingDataset,self).__init__()
        self.data = torch.Tensor(data).long()
        #self.data_freqs = torch.Tensor(word_freq)
        self.win_size = win_size
        self.neg_ratio = neg_ratio
        self.words,self.freqs = [],[]
        for word,freq in word_freq.items():
            self.words.append(word)
            self.freqs.append(freq/len(data))    
        #self.words = torch.Tensor(self.words).long()
        print(self.words)
        print(self.freqs)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        center_word = self.data[index]
        pos_indices = list(range(index - self.win_size, index)) + list(
            range(index + 1, index + self.win_size + 1))  # 中心词前后各C个词作为正样本
        pos_indices = list(filter(lambda i: i >= 0 and i < len(self.data), pos_indices))  # 过滤，如果索引超出范围，则丢弃
        pos_words = self.data[pos_indices]  # 周围单词
        # 按分布找出词典中的负样本，replace最好选True，不然窗口较大而字典较小的时候，数字就不够用了
        # 这个写法并不尽善尽美
        # 考虑一种情况：a跟b在前文距离较远，却在后文中很近，accdkvslbmnuvab
        # 这样，在前文遇到第一个a的时候可能把[a,b]当作负样本
        # 在走到后文的a的时候，却把[a,b]作为正样本对
        # 测试用例的数组里的31跟24模拟了这种情况
        neg_words = np.random.choice(a=self.words,size=(self.neg_ratio+1) * len(pos_words),\
            replace=True,p=self.freqs)
        neg_words = np.setdiff1d(neg_words,pos_words)[:self.neg_ratio * len(pos_words)]
        
        #neg_words = torch.multinomial(self.data_freqs, (self.neg_ratio+1) * pos_words.shape[0], True)
        return center_word,pos_words,torch.Tensor(neg_words).long()

if __name__=='__main__':
    data = [31,30,20,21,22,23,24,25,21,20,22,23,24,22,23,21,21,26,26,26,27,28,23,29,29,21,30,\
        31,24,24,24,24,24]
    word_freq = dict(collections.Counter(data))
    print(word_freq)
    wed = WordEmbeddingDataset(data=data,word_freq=word_freq)
    for i in range(len(wed)):
        center,pos,neg = wed.__getitem__(i)
        print('center=',center)
        print('pos=',pos)
        print('neg=',neg)


        




