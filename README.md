# DeepLearning
代码顺序
1. 完成SingleLayerPerceptron，练习单层感知机的实现
2. 完成LinearNN，实现线性神经网络，测试用例说明不能线性神经网络不能解决异或问题，但是特征做完非线性组合以后就可以。
3. 完成MultipleLayerPerceptron，实现多层感知机，测试用例说明加入非线性激活函数的多层感知机可以解决异或问题
4. 完成Graph和CalculateGraph，实现计算图，用python模拟pytorch中张量前向计算和后向计算的过程
5. 创建BaseDeepLearning文件夹存放前面所有基础深度学习入门的代码
6. 后面的代码需要服务器，所以尝试在服务器中使用git
7. 创建BasePytorch的notebook文件，练习基本的pytorch操作
8. 创建Word2Vec实现Skip-Gram和负采样
9. 补充NLP常用文字预处理过程
10. 添加SkipGramDataset，补充负采样的实现，文献3和文献4中负采样直接从词典中采集负样本，这可能导致词典中正样本被当作负样本，在代码中补充了负样本的条件筛选，这部分代码修改自文献4（文献4源代码也没有解决这个问题）
11. 添加SkipGram实现了skip gram的模型训练
12. 添加preNLP文件夹，复现前NLP时代的代码
13. SentimentClassificationDataset和data_preprocess文件内容比较杂


参考文献
1. 多层感知机以前的测试用例、可视化代码、数学推导来源于覃秉丰的《深度学习从0到1》
2. 计算图的基础代码和思路来源于张伟振的《深度学习原理与Pytorch实战》
3. pytorch基础和自然语言处理基础代码基本来自张楠等的《深度学习自然语言处理实战》
4. NLP文字预处理过程参考https://github.com/wentsun12/NLP_Learning/blob/master/skip_gram/skip-gram.py
5. skip-Gram模型和文字预处理参考https://zhuanlan.zhihu.com/p/275899732
6. 情感分类语料库测试代码主要来自：https://blog.csdn.net/m0_37201243/article/details/105609333