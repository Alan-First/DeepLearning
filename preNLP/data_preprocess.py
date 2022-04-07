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

path = './data'
if not os.path.exists(path):
    os.makedirs(path)
srcfile = os.path.join(path,'cornell_movie_dialogs_corpus.zip')
file = zipfile.ZipFile(srcfile,'r')
file.extractall(path)
corpus_file_list=os.listdir("./data/cornell movie-dialogs corpus")
print(corpus_file_list)
