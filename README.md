
## word2vec-MultiSense

### 简介

本代码fork自[google的word2vec](https://code.google.com/p/word2vec/) 代码，修改之后可以解决单词的一词多义的问题，得到单词的每个含义的词向量。例如：apple主要有两个含义，一个是吃的苹果，一个是苹果公司或者手机。

论文：[Efficient Non-parametric Estimation of Multiple Embeddings perWord in Vector Space](http://ciir-publications.cs.umass.edu/getpdf.php?id=1172)

### 如何使用


```
git clone https://github.com/wenjunoy/word2vec-MultiSense
cd word2vec-MultiSense/src/
make sense2vec_mssg
cd ../scripts/
./demo-sense-mssg.sh <corpus file> SENSE_K DIM #SENSE_K: sense的个数设置， DIM：向量的维度
```

对于非参模型，不需要设置sense的个数
```
cd word2vec-MultiSense/src/
make sense2vec_np
cd ../scripts/
./demo-sense-np.sh <corpus file> DIM #DIM：向量的维度
```

得到3个文件，如下：

#### 单词的全局向量 ***.vec
```
50000 100
apple 0.01 0.32 ...
```
#### 单词每个含义向量 ***.vec.sense
```
250000 100
...
apple_0 0.143 0.032...
apple_1 0.123 0.3214...
apple_2
apple_3
apple_4
```
#### 单词的每个含义在语料库中出现的次数 ***.vec.sense.num
```
apple_0 4
apple_1 2
apple_2 5
apple_3 221338
apple_4 246205
```
从每个含义出现的次数可以看出，单词apple主要有两种含义（吃的水果和苹果手机）。我们从向量的最近邻也可以来验证：

apple_3的最近邻单词：
![](http://7xlx99.com1.z0.glb.clouddn.com/git/20171023105121.png)

apple_4的最近邻单词：
![](http://7xlx99.com1.z0.glb.clouddn.com/git/20171023105215.png)
