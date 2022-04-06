#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :SkipGram.py
@Description  :搭建带有负采样的跳字模型数据集，数据库代码来自文献4，功能上做了优化，测试用例来自文献5
@Time         :2022/04/02 21:32:45
@Author       :Hedwig
@Version      :1.0
'''

import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.data as tud
import collections
    
# 这部分代码来自文献4，但是源代码没有考虑负采样的时候，负样本不能是正样本这种情况，
# 而是直接从词典里按频率分布取了数，所以这里修改了一下它的实现
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self,data,win_size=1,neg_ratio=2,word_freq=[]):
        super(WordEmbeddingDataset,self).__init__()
        self.data = torch.Tensor(data).long()
        self.win_size = win_size
        self.neg_ratio = neg_ratio
        self.words,self.freqs = [],[]
        for word,freq in word_freq.items():
            self.words.append(word)
            self.freqs.append(freq)
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
        # 目前没有想到解决方案，但是网上的实现也都不考虑这个问题，原因不明
        neg_words = np.random.choice(a=self.words,size=(self.neg_ratio+1) * len(pos_words),\
            replace=True,p=self.freqs)
        neg_words = np.setdiff1d(neg_words,pos_words)[:self.neg_ratio * len(pos_words)]
        neg_words = [self.words.index(i) for i in neg_words]
        return center_word,pos_words,torch.LongTensor(neg_words)

def get_word_freq(data,FREQ=0,VOCABULARY_SIZE=50000,DELETE_WORDS=False):
    """
    @Params :文字序列
    
    @Description :本代码主要来自文献5
    
    @Returns     :词频字典
    """
    # 取出频数前VOCABULARY_SIZE的单词作为词典
    counts_dict = dict((collections.Counter(data).most_common(VOCABULARY_SIZE-1)))
    # 去掉频数小于 FREQ 的单词，这些单词最后都被标记为UNK
    # trimmed_words = [word for word in data if counts_dict[word] > FREQ]
    # 计算 UNK 的频数 = 单词总数 - 前 50000 个单词的频数之和
    counts_dict['UNK']=len(data)-np.sum(list(counts_dict.values()))
    #建立词和索引的对应
    idx_to_word = [word for word in counts_dict.keys()]# 把词典所有词都记录下来
    word_to_idx = {word:i for i,word in enumerate(idx_to_word)}# 词语-idx对
    data = [word_to_idx.get(word,word_to_idx["UNK"]) for word in data]# 一段文字变成一段idx句子，列表生成式
    # 计算单词频次
    total_count = len(data)
    word_freqs = {w: c/total_count for w, c in counts_dict.items()}
    # 以一定概率去除出现频次高的词汇
    if DELETE_WORDS:
        t = 1e-5
        prob_drop = {w: 1-np.sqrt(t/word_freqs[w]) for w in data}
        data = [w for w in data if np.random.random()<(1-prob_drop[w])]
    else:
        data = data
    # 计算词频,按照原论文转换为3/4次方
    # word_counts = np.array([count for count in counts_dict.values()],dtype=np.float32)
    # word_freqs = word_counts/np.sum(word_counts)
    # word_freqs = word_freqs ** (3./4.)
    word_freqs = {w: np.round(c**(3./4.),4) for w, c in counts_dict.items()}
    word_sum = np.sum(list(word_freqs.values()))
    word_freqs = {w: c/word_sum for w, c in word_freqs.items()}
    return data,word_freqs
if __name__=='__main__':
    VOCABULARY_SIZE=10# 词典大小
    data = [31,30,20,21,22,23,24,25,21,20,22,23,24,22,23,21,21,26,26,26,27,28,23,29,29,21,30,\
        31,24,24,24,24,24]
    data = [str(i) for i in data]# 模拟一段str文字数组
    #word_freq = dict(collections.Counter(data))
    id_word,word_freq = get_word_freq(data,VOCABULARY_SIZE)
    print(word_freq)
    print(np.sum(list(word_freq.values())))
    wed = WordEmbeddingDataset(data=id_word,word_freq=word_freq)
    for i in range(len(wed)):
        center,pos,neg = wed.__getitem__(i)
        print('center=',center)
        print('pos=',pos)
        print('neg=',neg)
    

        




