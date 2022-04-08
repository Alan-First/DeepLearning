#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :data_preprocess.py
@Description  :下载解压数据集
@Time         :2022/04/08 00:13:38
@Author       :Hedwig
@Version      :1.0
'''




# 数据集是Cornell Movie-Dialogs Corpus
# 下载链接："http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"

# 数据集解压
import os
import time
import zipfile


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

path = './data/'
if not os.path.exists(path):
    os.makedirs(path)
srcfile = os.path.join('cornell_movie_dialogs_corpus.zip')
file = zipfile.ZipFile(srcfile,'r')
file.extractall(path)
corpus_file_list=os.listdir("./data/cornell movie-dialogs corpus")
print(corpus_file_list)
# 查看文件内容，各个文件含义可以在这个链接看到：https://cloud.tencent.com/developer/article/1747520
for file_name in corpus_file_list:    
    file_dir = os.path.join("./data/cornell movie-dialogs corpus", file_name)
    print(file_dir,"的前10行")
    printLines(file_dir)

# 关键是其中movie_lines.txt文件
# 以上代码都来自文献6，但是这份数据集本身没有人工的情感标注，所以后面操作没有继续
# 有兴趣可以从文献6链接继续跟进
# 人工情感分类的代码用的数据是另外下载的rt-polarity.neg和rt-polarity.pos