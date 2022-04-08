#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :load_param.py
@Description  :用于加载模型参数，代码来自文献3
@Time         :2022/04/07 23:28:42
@Author       :Hedwig
@Version      :1.0
'''

from distutils import text_file
import os
import argparse
import datetime
import torch
import torchtext.legacy.data as data

def load_param():
    parser = argparse.ArgumentParser(description='TextCNN classifier')
    parser.add_argument('-train-file',type=str,default='')
    parser.add_argument('-dev-file',type=str,default='')
    parser.add_argument('-test-file',type=str,default='')
    parser.add_argument('-min-freq',type=int,default=1,help='')
    parser.add_argument('-embed-dim',type=int,default=128)
    parser.add_argument('-kernel-num',type=int,default=100)
    parser.add_argument('-max-norm',type=float,default=5.0)
    parser.add_argument('-dropout',type=float,default=0.5)
    parser.add_argument('-cuda',type=bool,default=True)
    parser.add_argument('-device_id',type=int,default=0)
    args = parser.parse_args()
    return args




def load_data(config):
    # Field可以用于初始化数据
    # TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), init_token='<SOS>', eos_token='<EOS>',lower=True)
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)


if __name__=='__main__':
    config = load_param()
    if config.cuda is True:
        print('using GPU to train...')
        torch.cuda.manual_seed_all(2022)
        torch.cuda.set_device(config.device_id)
    load_data(config)
