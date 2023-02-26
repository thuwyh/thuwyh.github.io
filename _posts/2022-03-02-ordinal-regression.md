---
layout: single
title: 分级利器：序数回归
date:   2022-03-02 10:25:03 +0800
categories: 
    - 深度学习
tags: 
    - 序数回归
    - "Ordinal Regression"
    - 分类
    - 回归
comments: true
toc: true
header:
    teaser: /assets/sigmoid.png
---

之前面试的时候遇到过好几个候选人在做”评级“相关的项目，例如针对图片、视频的色情程度，评论的粗鲁程度分级等等。绝大部分的人在面对这种问题时通常想到的都是用**回归**或**分类**来解决。这两种方法都是有效的，但都有一些问题：

1. 常规的分类无法很好地建模类别间的关系。例如你要对评论的不文明程度分5档，但对于分类常用的交叉熵损失函数来说，把一个最高档的评论分成了最低档还是中间档对它来说损失是一样的。但对于实际的业务，前者显然比后者是更难接受的。
2. 回归算法需要比较多的超参调试。在[之前的文章里](https://mp.weixin.qq.com/s?__biz=MzI4MzEyOTIzOA==&mid=2648563919&idx=1&sn=aab7b155c29866f63dacb8fe9015c638&chksm=f3a62436c4d1ad205cc069d02ac8a17e8b2d9c4c92217610926ae612918415795f073193f221&token=509546952&lang=zh_CN#rd)聊过，回归对标签的尺度是敏感的，把细粒度，例如100档（标签为1-100）的评级问题直接交给MSE Loss往往得不到好的结果。回归对标签中的最大值和最小值也天然会有一些抗拒。

在[Pawpularity比赛](https://www.kaggle.com/c/petfinder-pawpularity-score/overview, "Pawpularity比赛")结束后[CPMP](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/300942, "CPMP的帖子")说他使用了一种叫**Ordinal Regression**（中文名没找到，姑且称它为序数回归）的方法，我在一些评级问题上尝试之后发现这个方法确实行之有效，而且非常简单优美。

### 数学解释

说是序数“回归”，但我它认为本质上是一个**考虑了类别间关系的分类算法。** 大家一定都很熟悉sigmoid函数$σ$，它的定义域是(-∞,+∞)，而值域是(0,1)，且是单调增函数，连续可导。我们可以把$σ(x)$看做是随机变量小于x的概率，即某个(-∞,+∞)上随机变量的累积分布函数（CDF）。

![Sigmoid函数](/assets/sigmoid.png)

假设我要处理一个5档的分类问题，而上面说的随机变量就是模型的输出，那么问题可以转化为找到四个**切分点**$\theta_1, \theta_2, \theta_3, \theta_4$，并用$P(x<\theta_1)$, $P(\theta_1< x<\theta_2)$, $P(\theta_2< x<\theta_3)$, $P(\theta_3< x<\theta_4)$, $P(\theta_4< x<+\infty)$这五个概率来表示$x$分别属于五个等级的概率。进一步结合前面的sigmoid函数做CDF的方法，可以把五个概率转化为$σ(\theta_1-x)$, $σ(\theta_2-x)-σ(\theta_1-x)$, $σ(\theta_3-x)-σ(\theta_2-x)$, $σ(\theta_4-x)-σ(\theta_3-x)$, $1-σ(\theta_4-x)$。

这样我们就把一个模型输出的实数logit转化成了属于五个等级的概率，进而可以使用负对数似然损失函数来优化这个分类问题。在这样的设定下既可以使用一组固定的切分点来优化模型，又可以把切分点也作为可学习的权重和模型一起优化。

### 代码

一开始我在网上找到了一个pytorch的Ordinal Regression实现[spacecutter](https://github.com/EthanRosenthal/spacecutter)，但经过一番实验之后我发现它写的并不完美，于是自己又修改了一下，在这里分享给大家

```python
class OrdinalRegressionLoss(nn.Module):

    def __init__(self, num_class, train_cutpoints=False, scale=20.0):
        super().__init__()
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float()*scale/(num_class-2) - scale / 2
        self.cutpoints = nn.Parameter(self.cutpoints)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred, label):
        sigmoids = torch.sigmoid(self.cutpoints - pred)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        neg_log_likelihood = torch.log(likelihoods)
        if label is None:
            loss = 0
        else:
            loss = -torch.gather(neg_log_likelihood, 1, label).mean()
            
        return loss, likelihoods
```

主要的改动是：

- 增加了scale参数。这个参数用来控制初始切分点的稀疏程度。我发现让切分点稀疏些可以提高这个loss的性能，特别是在档位很多的时候；但是神经网络不擅长输出很大的值，因此又不能太稀疏。这是个主要的待调超参数。
- 初始化cutpoints的时候间距计算更加精确。原来的代码是像下面这样，若有4个切分点（5类），则初始化出来的参数为[-2, -1, 0, 1]，不对称。

```python
num_cutpoints = self.num_classes - 1
cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
```

- 增加了参数训练开关`train_cutpoints`。因为我发现有时fix切分点得到的结果更好。
- 增加了`likelihoods`输出，使得该模块不仅能用于训练求loss，也能用于推理求类别。

其实看代码很容易搞懂序数回归到底是在做什么。也很容易看出它通过概率差在切分点间建立起的联系，这在普通的分类（输出多类logit，并用softmax综合）里是没有的，也是它先进的地方。


希望这篇文章对你有帮助，有任何问题欢迎在文末留言，也可以关注公众号添加我的微信交流。特别感谢Z by HP对我的支持。

>As a Z by HP Global Data Science Ambassador, Yuanhao's content is sponsored and he was provided with HP products.