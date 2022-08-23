# nlp大作业---NMT (neural machine translation)

机器翻译( zh-en ）Seq2seq

##	项目背景

机器翻译, 又称自动翻译, 是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程. 它是人工智能领域一个重要的部分. 最早的机器翻译是基于固定规则和词典的方法, 即基于规则的方法, 但是这种方法受限于规则, 具有较大的局限性. 后来出现了基于统计的统计机器翻译, 使用统计学的方法翻译输入的句子, 效果比之前有较大提升. 而近年来, 随着人工神经网络技术的发展, 使用人工神经网络进行机器翻译的神经机器翻译(NMT)逐渐兴起, 翻译质量得到极大提升, 应用也逐渐广泛, 比如著名的Google translate就是使用的这种技术. 
![image](https://user-images.githubusercontent.com/60745334/186149557-e16a7735-de69-4bda-9996-8ad2c1a43d2f.png)

##	使用Attention机制的Seq2Seq

Seq2Seq是一个Encoder-Decoder结构的网络, 它的输入是一个序列, 输出也是一个序列, 其中Encoder中将一个可变长度的序列变为固定长度的向量表达, Decoder将这个固定长度的向量变为可变长度的目标序列.

![image](https://user-images.githubusercontent.com/60745334/186149636-b7048696-1e43-4eac-89f6-300bddaa943e.png)

在这个模型中, 使用的是循环神经网络(RNN), 所以会有梯度消失问题, 为了解决这个问题, 我们引入了注意力机制, 可以让模型只注意输入相关的部分. 

![image](https://user-images.githubusercontent.com/60745334/186149676-fa7446a6-0772-438b-a4d8-48879827df28.png)


在Encoder中, 句子的index向量经过一个embedding层, 生成一个词向量, 之后将这个词向量与hidden一起送入GRU中, 得到output和hidden.

![image](https://user-images.githubusercontent.com/60745334/186149713-baf77795-aeb3-412b-8707-40e64c1c9ae9.png)

在Decoder中, 首先将上一轮Decoder的output传入作为input(初始为<sos>), 将此input通过一个embedding层. 之后与encoder的hidden结合, 通过attention层, 归一化后形成注意力权重. 之后将此注意力权重分配到encoder的output上, 实现注意力机制, 得到attn_applied, 之后将attn_applied和上一轮的输出结合, 并通过relu层激活, 最后将此结果与hidden一起送入GRU中, 得到output和hidden, 此output便是输出单词的概率分布. 
  
  ![image](https://user-images.githubusercontent.com/60745334/186149759-2d2126dc-e0dd-40eb-949c-8aa145b39653.png)

## 实验
  
  按照98:1:1的比例划分了训练集, 评估集和测试集
模型在训练集上训练, 损失函数使用交叉熵. 训练N次后计算在验证集上的bleu的平均得分, 如果比历史最高记录要高, 则覆盖保存. 

实验结果如下:
loss曲线如下, 可以看出模型在逐渐收敛

  ![image](https://user-images.githubusercontent.com/60745334/186150113-a561be99-5203-4d2c-8a9c-285d0174b533.png)

测试集上的bleu平均得分为0.2596, 分布如下:
  
  ![image](https://user-images.githubusercontent.com/60745334/186150158-bbabd2fa-f7d9-47cb-8d80-e570cc180d03.png)

 输出效果对比:
一些比较好的结果:

  ![image](https://user-images.githubusercontent.com/60745334/186150198-7efcff1a-5ea4-4aa8-890e-90bf96bcd186.png)
![image](https://user-images.githubusercontent.com/60745334/186150212-64bcf1a5-b133-4b8c-89e0-dcd4f3cb88d2.png)
![image](https://user-images.githubusercontent.com/60745334/186150222-66886684-a673-4cc8-974d-4d823089e31a.png)

  一些较差的结果:
  
  ![image](https://user-images.githubusercontent.com/60745334/186150259-43d7bd38-e9b7-41b7-8a32-bbaeacefa86d.png)

  可以看出存在漏词和重复结尾标点的问题. 
  
## 项目结构
  
config.py 为配置文件
data_processing.py是提供一些处理文本的类和函数, 比如生成index向量
model.py是Seq2Seq网络的结构
train.py是训练函数, 用于训练, 训练方法为python train.py
evaluate.py提供了一些用于评估的函数和类
test.ipynb 用于测试训练好的模型效果










