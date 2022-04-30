#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :GPT2Test.py
@Description  :GPT2预测词语
@Time         :2022/04/30 06:00:26
@Author       :Hedwig
@Version      :1.0
'''

import torch
from transformers import GPT2Tokenizer,GPT2LMHeadModel
# 加载预训练模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')#GPT2没有为文本添加标识符
#输入编码
indexed_tokens = tokenizer.encode("Who is Li Jinhong ? Li Jinhong is a")
print(tokenizer.decode(indexed_tokens))
tokens_tensor = torch.tensor([indexed_tokens])
# 加载预训练模型
model = GPT2LMHeadModel.from_pretrained('gpt2')
# 设置模型为评估模式
model.eval()
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

with torch.no_grad():
    outputs = model(tokens_tensor)#outputs是一个实例对象，predictions是一个tensor
    predictions = outputs[0]
predicted_index = torch.argmax(predictions[0,-1,:]).item()# [1,句子长度,词表长度]
predicted_text = tokenizer.decode(indexed_tokens+[predicted_index])
print(predicted_text)
# 输出：Who is Li Jinhong? Li Jinhong is a young

# 如果想用手动方式载入token，要执行以下两步
# 1} tokenizer = GPT2Tokenizer('gpt2-vocab.json文件路径','gpt2-merges.txt文件路径')
# 如果想使用from_trained方法载入，那么gpt2-vocab.json要重命名为vocab.json，同时
# gpt2-merges.txt文件要改名为merges.txt，然后在from_pretrained传入gpt2文件夹路径即可
# 2} model = GPT2LMHeadModel.from_pretrained('bin模型路径'，config='gpt2-config.json文件') 

# 循环方式不断调用gpt2生成词直到句子完整
stopids = tokenizer.convert_tokens_to_ids(["."])[0]
print(stopids)
#print(indexed_tokens)
past = None
for i in range(100):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]# 
    predicted_index = torch.argmax(predictions[0,-1,:]).item()
    #print(predicted_index) #取出
    indexed_tokens += [predicted_index]#
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    if stopids == predicted_index:
        break

sequence = tokenizer.decode(indexed_tokens)
print(sequence)



