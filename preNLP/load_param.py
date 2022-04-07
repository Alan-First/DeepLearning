#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :load_param.py
@Description  :用于加载模型参数，代码来自文献3
@Time         :2022/04/07 23:28:42
@Author       :Hedwig
@Version      :1.0
'''


import argparse

def param():
    parser = argparse.ArgumentParser(description='TextCNN classifier')
    parser.add_argument('-train-file',type=str,default='')
    parser.add_argument('-dev-file',type=str,default='')
    parser.add_argument('-test-file',type=str,default='')
    parser.add_argument('-min-freq',type=int,default=1,help='')
    parser.add_argument('-embed-dim',type=int,default=128)
    parser.add_argument('-kernel-num',type=int,default=100)
    parser.add_argument('-max-norm',type=float,default=5.0)
    parser.add_argument('-dropout',type=float,default=0.5)