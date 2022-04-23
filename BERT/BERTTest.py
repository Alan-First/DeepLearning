#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :BERTTest.py
@Description  :
@Time         :2022/04/11 20:58:20
@Author       :Hedwig
@Version      :1.0
'''

from unittest.mock import sentinel
from sklearn import preprocessing
import torch
from torch import dtype
from torch import embedding
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer,BertModel
from torchtext.legacy import data,datasets
import numpy as np
import random
import time


SEED = 2022
TRAIN = True
BATCH_SIZE=128
N_EPOCHS=20
HIDDEN_DIM=256
OUTPUT_DIM=1
N_LAYERS=2
BIDIRECTIONAL=True
DROPOUT=0.25
TEXT = 'I like you'

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 情感分类任务在bert加上两层gru
class SentimentModel(nn.Module):
    def __init__(self,bert,hidden_dim,output_dim,n_layers,bidirectional,dropout) -> None:
        super(SentimentModel,self).__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers<2 else dropout
            )
        self.out = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim,output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _,hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        output = self.out(hidden)
        return output

model = SentimentModel(bert_model,HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT)
#print(model)

def epoch_time(start_time,end_time):
    elapsed_time = end_time-start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time-(elapsed_mins*60))
    return elapsed_mins,elapsed_secs

def binary_accuracy(preds,y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()
    acc = correct.sum()/len(correct)
    return acc

def train(model,iterator,optimizer,criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions,batch.label)
        acc = binary_accuracy(predictions,batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator)

def evaluate(model,iterator,criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions,batch.label)
            acc = binary_accuracy(predictions,batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator)

def predict_sentiment(model,tokenizer,sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_len-2]
    indexed = [init_token_id]+tokenizer.convert_tokens_to_ids(tokens)+[eos_token_id]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

if __name__=='__main__':
    if TRAIN:
        train_iter,valid_iter,test_iter = load_data()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss().to(device)
        model = model.to(device)
        best_val_loss = float('inf')
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss,train_acc = train(model,train_iter,optimizer,criterion)
            valid_loss,valid_acc = evaluate(model,valid_iter,criterion)
            end_time = time.time()
            epoch_mins,epoch_secs = epoch_time(start_time,end_time)
            if valid_loss<best_val_loss:
                best_val_loss = valid_loss
                torch.save(model.state_dict(),'model.pt')
            print(f'Epoch:{epoch+1:02}|Epoch Time:{epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss:{train_loss:.3f}|Train Acc:{train_acc*100:.2f}%')
            print(f'\t Val Loss:{valid_loss:.3f}| Val Acc:{valid_acc*100:.2f}%')
        model.load_state_dict(torch.load('model.pt'))
        test_loss,test_acc = evaluate(model,test_iter,criterion)
        print(f'Test Loss:{test_loss:.3f}|Test Acc:{test_acc*100:.2f}%')
    else:
        model.load_state_dict(torch.load('model.pt',map_location=device))
        sentiment = predict_sentiment(model,tokenizer,TEXT)
        print(sentiment)

'''
Epoch:01|Epoch Time:7m 14s
        Train Loss:0.569|Train Acc:69.34%
         Val Loss:0.377| Val Acc:83.68%
Epoch:02|Epoch Time:7m 20s
        Train Loss:0.378|Train Acc:83.65%
         Val Loss:0.276| Val Acc:88.91%
Epoch:03|Epoch Time:7m 21s
        Train Loss:0.320|Train Acc:86.45%
         Val Loss:0.264| Val Acc:89.35%
Epoch:04|Epoch Time:7m 20s
        Train Loss:0.291|Train Acc:87.99%
         Val Loss:0.260| Val Acc:90.11%
Epoch:05|Epoch Time:7m 20s
        Train Loss:0.269|Train Acc:88.85%
         Val Loss:0.251| Val Acc:90.07%
Test Loss:0.237|Test Acc:90.51%
'''
