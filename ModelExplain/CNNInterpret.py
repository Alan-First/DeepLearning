#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :CNNInterpret.py
@Description  :模型可解释性
@Time         :2022/04/30 19:39:44
@Author       :Hedwig
@Version      :1.0
'''

#from turtle import forward
import spacy
import torch
#from torch import embedding
import torch.nn.functional as F
from captum.attr import (
    IntegratedGradients,TokenReferenceBase,visualization,configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer
)
from TextCNN import TextCNN,TEXT,LABEL

# Captum是一个基于pytorch的模型可解释库，它实现了当今主流神经网络可解释性算法：如集成梯度、深度弯曲
# 和传导等。它有一个可视化工具Captum Insights，目前只支持集成梯度
# 梯度积分：输入数据特征从无到有的过程中，计算每个阶段的梯度并将它们加起来
# 在NLP中，输入词id先经过词嵌入向量表将词转为词嵌入，词嵌入送入神经网络更新网络参数
# 在分析词嵌入模型时，一般会将第一步隔离起来，直接对映射好的词嵌入做梯度积分
# captum中有一个configure_interpretable_embedding_layer可以将原有模型的词嵌入层包装起来，并支持
# 对词嵌入权值单独提取，用它配合梯度积分算法能实现词嵌入的可解释性
  
class TextCNNInterpret(TextCNN):
    def __init__(self,*args,**kwargs):# 透传参数
        super().__init__(*args,**kwargs)
    def forward(self,text):
        # 删除了父类中调换维度的部分，这样的模型是不能直接运行的
        embedded = self.embedding(text)# 从词嵌入开始处理
        # 以下与父类一样
        embedded = embedded.unsqueeze(1) #形状为[batch size, 1, sent len, emb dim]
        #len(filter_sizes)个元素，每个元素形状为[batch size, n_filters, sent len - filter_sizes[n] + 1]
        conved = [self.mish(conv(embedded)).squeeze(3) for conv in self.convs]
        #len(filter_sizes)个元素，每个元素形状为[batch size, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))#形状为[batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

# d定义模型参数
INPUT_DIM = len(TEXT.vocab)#25002
EMBEDDING_DIM = TEXT.vocab.vectors.size()[1]#100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
#实例化模型
model = TextCNNInterpret(
    INPUT_DIM,EMBEDDING_DIM,N_FILTERS,FILTER_SIZES,OUTPUT_DIM,DROPOUT,PAD_IDX
)
model.load_state_dict(torch.load('textcnn-model.pt'))
print('Vocabulary Size:',len(TEXT.vocab))
# 提取词嵌入层的部分，调用configure_interpretable_embedding_layer函数
# 参数是指定的模型对象和模型嵌入层，该函数根据这些信息提取嵌入层并封装 
interpretable_embedding = configure_interpretable_embedding_layer(model,'embedding')

ig = IntegratedGradients(model)# 创建梯度积分对象
vis_data_records_ig = []# 定义列表，存放可视化记录
nlp = spacy.load('en_core_web_sm')# 为分词库加载英文语言包
def interpret_sentence(model,sentence,min_len=7,label=0):
    sentence = sentence.lower() #句子转小写
    model.eval()
    # 分词处理
    text = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(text)< min_len:
        text += [TEXT.pad_token]*(min_len-len(text))
    # 句子中的单词转索引
    indexed = [TEXT.vocab.stoi[t] for t in text]
    model.zero_grad()
    input_indices = torch.LongTensor(indexed)#转张量
    input_indices = input_indices.unsqueeze(0)#添加维度
    input_embedding = interpretable_embedding.indices_to_embeddings(input_indices)
    # 词嵌入输入模型预测
    pred = torch.sigmoid(model(input_embedding)).item()
    pred_ind = round(pred)

    # 创建梯度积分的初始输入值，创建与输入等长的pad数组转为词向量
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
    # 制作初始输入索引：复制指定长度个token_reference，扩展维度
    reference_indices = token_reference.generate_reference(len(indexed),device='cpu').unsqueeze(0)
    print('reference_indices',reference_indices)
    # 输入索引转词嵌入
    reference_embedding = interpretable_embedding.indices_to_embeddings(reference_indices)
    # 梯度积分计算可解释性
    # 将"从全pad词向量，到输入句子词向量"，均匀分500份，依次放到模型，求出相邻分数的梯度再求和
    attribute_ig,delta = ig.attribute(
        input_embedding,reference_embedding,n_steps=500,return_convergence_delta=True
    )
    print('attribute_ig',attribute_ig.size(),delta.size())
    print('pred',LABEL.vocab.itos[pred_ind],'(','%.2f'%pred,')','delta:',abs(delta))
    # 加入可视化记录
    add_attributions_to_visualize(attribute_ig, text, pred, pred_ind, label, delta, vis_data_records_ig)
# 解释性结果加入可视化记录
def add_attributions_to_visualize(attributions,text,pred,pred_ind,label,delta,vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions/torch.norm(attributions)
    attributions = attributions.detach().numpy()
    # 结果添加到表里
    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,
        pred,
        LABEL.vocab.itos[pred_ind],
        LABEL.vocab.itos[label],
        LABEL.vocab.itos[1],
        attributions.sum(),
        text[:len(attributions)],
        delta))

# 输入句子测试
interpret_sentence(model, 'It was a fantastic performance !', label=1)

interpret_sentence(model, 'The film is very good！', label=1)

interpret_sentence(model, 'I think this film is not very bad！', label=1)


#根据可视化记录生成网页
visualization.visualize_text(vis_data_records_ig)

#还原模型的词嵌入层
remove_interpretable_embedding_layer(model, interpretable_embedding)
'''
训练数据集 17500 条
验证数据集 7500 条
测试数据集 25000 条
Mish activation loaded...
Vocabulary Size: 25002
/mistgpu/site-packages/captum/attr/_models/base.py:189: UserWarning: In order to make embedding layers more interpretable they will be replaced with an interpretable embedding layer which wraps the original embedding layer and takes word embedding vectors as inputs of the forward function. This allows us to generate baselines for word embeddings and compute attributions for each embedding dimension. The original embedding layer must be set back by calling `remove_interpretable_embedding_layer` function after model interpretation is finished. 
  "In order to make embedding layers more interpretable they will "
reference_indices tensor([[1, 1, 1, 1, 1, 1, 1]])
attribute_ig torch.Size([1, 7, 100]) torch.Size([1])
pred pos ( 0.98 ) delta: tensor([0.0008], dtype=torch.float64)
reference_indices tensor([[1, 1, 1, 1, 1, 1, 1]])
attribute_ig torch.Size([1, 7, 100]) torch.Size([1])
pred pos ( 0.88 ) delta: tensor([0.0001], dtype=torch.float64)
reference_indices tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
attribute_ig torch.Size([1, 9, 100]) torch.Size([1])
pred neg ( 0.43 ) delta: tensor([0.0002], dtype=torch.float64)
<IPython.core.display.HTML object>
'''