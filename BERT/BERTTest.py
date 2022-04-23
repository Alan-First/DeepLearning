#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :BERTTest.py
@Description  :
@Time         :2022/04/11 20:58:20
@Author       :Hedwig
@Version      :1.0
'''

from sklearn import preprocessing
import torch
from torch import dtype
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer,BertModel
from torchtext.legacy import data,datasets
import numpy as np
import random
import time


SEED = 2022
TRAIN = False
BATCH_SIZE=128
N_EPOCHS=5
HIDDEN_DIM=256
OUTPUT_DIM=1
N_LAYERS=2
BIDIRECTIONAL=True
DROPOUT=0.25

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
device='cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
init_token_id = tokenizer.cls_token_id
eos_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id
unk_token_id = tokenizer.unk_token_id

# 句子最长长度
max_input_len = tokenizer.max_model_input_sizes['bert-base-uncased']

#将句子长度切成510长，加上开头和结尾token
def tokenize_and_crop(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_len-2]
    return tokens

# 加载pytorch提供的IMDB数据
def tokenize_and_crop(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_len-2]
    return tokens

def load_data():
    text = data.Field(
        batch_first=True,
        use_vocab=False,
        tokenize=tokenize_and_crop,
        preprocessing=tokenizer.convert_tokens_to_ids,
        init_token=init_token_id,
        pad_token=pad_token_id,
        unk_token=unk_token_id
    )
    label = data.LabelField(dtype=torch.float)
    train_data,test_data = datasets.IMDB.splits(text,label)
    print(train_data)
    train_data,valid_data = train_data.split(random_state=random.seed(SEED))
    print(f'train examples counts:{len(train_data)}')
    print(f'test examples counts:{len(test_data)}')
    print(f'valid examples counts:{len(valid_data)}')

    label.build_vocab(train_data)

    train_iter,valid_iter,test_iter = data.BucketIterator.splits(
        (train_data,valid_data,test_data),
        batch_size=BATCH_SIZE,
        device=device
        )
    return train_iter,valid_iter,test_iter


