# Word2Vec
## 词向量编码
1. 自然语言处理最先的一步是要将文字转换成机器可以识别和计算的数字，也就是词向量编码
2. 文本表示分为离散表示（独热编码、词袋模型等）和分布式表示，分布式表示包括词嵌入（word embedding）和共现矩阵（co-occurrence Matrix）等，词嵌入就包括word2vec、Glove等
3. 最开始采用的是独热编码(one-hot)，但是词汇表的扩大极易形成高维稀疏向量，带来维度灾难，同时无法衡量词语之间的相关性
4. 目前采用的word2vec包括跳字模型（Skip-Gram）和连续词袋模型（CBOW）
5. 为了解决词袋过大导致词向量编码模型输出维度过大的问题，主要采用层次softmax和负采样的方法

## Skip-Gram
1. 使用负采样Skip-Gram的实现思路是这样的：给定两个embedding层，一个用于将输入的中心词压缩编码，另一个用于将背景词跟负样本压缩编码，然后将压缩中心词跟压缩背景词和压缩负样本词分别计算内积，构造一个损失函数用于尽量增大前者的内积而降低后者的内积，本代码实现的时候把两个embedding层都用同一个，因为其中一个embedding是该词作为中心词以后的映射，另一个是作为背景词的映射，这两种是对称的，同时本代码十使用的压缩中心词与压缩背景词都是[batch_size,embedding]，也就是说每次输入不是单中心词多个背景词输入，而是一个中心词一个背景词输入，其实后面多次输入同一个中心词和不同的背景词，也是可以达到单中心词多背景词的效果。
2. 