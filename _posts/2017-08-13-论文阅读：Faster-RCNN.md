Faster-RCNN分为两个stage，第一个阶段是Region Proposal Network(RPN)，接收一张
图片并生成Region Proposal并喂给第二阶段的Fast-RCNN，分别开看一下每个部分是如何
工作的。

**本文由本博主原创，版权归本博主所有，转载请注明出处，否则将依法追究责任**

---

# Region Proposal Network(RPN)

RPN的主要思想是先在原始图像上的不同位置人为摆放大量的不同大小不同纵横比的bounding boxes(或者叫anchors)，
只要放的足够多，总有一些bounding boxes是能够与ground truth的bounding boxes有比较大的
重叠(iou)的，因此需要一个网络（也就是RPN）来告诉我们哪些人为放的bounding boxes与ground truth
的bounding boxes是有大量重叠的，哪些没有，并告诉我们位置（坐标）上还差多少就能够完全重叠(偏移量)。



以VGG16（网络结构如下）为base network为例


![vgg16](http://i4.bvimg.com/604224/4450e2b58b38c5fe.png)

从第五个pooling层（包括）截断，这样一共有四个pooling层，图片尺寸变为原来的十六分之一。
以一张640 * 960的图像为例，生成的feature map的大小为`[40, 60, 512]`。

---

## anchors
那么在图像的哪些位置放多大尺寸多少纵横比的bounding boxes(anchors)呢？作者的选择是在这个feature map
的每个位置上放9个大小纵横比都不同的anchors（anchor的中心与每个位置对齐），然后将这9个anchors映射回原来的图像上（映射方式为feature map的(i, j)回到图像的(16i, 16j)）。
这9个anchors取了3种尺度(`[128, 256, 512]`)，3中纵横比(`[1/2, 1, 2]`)，见下图

![anchros_1](http://i4.bvimg.com/604224/bbc5a450db49c753.png)

对于`[40, 60, 512]`的feature map来说就会生成`40 * 60 * 9`这么多anchors，为了确定这些anchors
哪些是与ground truth的bounding boxes重合（或者说anchor包含object）的，每个anchors需要输出一个
二维的概率向量表示是否包含object，还需要4个值用来对anchor的位置进行微调使之与ground truth的bounding
boxes完全重合。

因此对于feature map`40 * 60`的每一个位置我们需要`9`个二维的概率向量(`9 * 2`)和`9`个四维的向量(`9 * 4`)来最终确定哪些anchors有用。
一共需要`9 * 6 = 54`个输出值。

---

## 子网络subnetworks
既然要对feature map每一个位置都做这件事情，一个合理的想法就是把`[40, 60, 512]`的每一个位置的向量也就是`[1, 1, 512]`用全连接层映射到54维
的输出上，但是这样做有一个不好的地方在于，只有spatial位置上的1所包含的respect field（感受野）太小了，不能完全决定这个anchor是否有用，因此
作者选择了`[3, 3, 512]`这样的输入，spatial位置上有3个单位，感受野能大不少。

因此为了在每个位置上生成54维的输出，作者将`[40, 60, 512]`以每个位置为中心的`[3, 3, 512]`的特征拿出来，送入一个中间层（维度取成了512），最后
这个中间层再映射到54维的输出上，也就相当于`[3, 3, 512, 512]`的卷积核作用在`[40, 60, 512]`的中间层特征得到`[40, 60, 512]`，然后用`[1, 1, 512, 54]`生成`[40, 60, 54]`的输出，最后切割成
`[40, 60, 9, 2]`和`[40, 60, 9, 4]`两个输出，前者表示40 * 60的每个位置上的9个anchors中每个anchor包含object的概率，后者表示每个anchor距离真实的bounding box
的坐标的偏移。

---

## 损失函数
对于每一个anchor，需要知道anchor是否真的包含一个object，具体的过程为，一个anchor将被
标记为positive的（也就是包含object），当且仅当如下条件满足之一：
* 与任何一个ground truth的bounding box的iou大于0.7
* 与任何一个ground truth的bounding box的iou比其他任何一个的anchor都大

有了label，RPN的损失函数的定义为:

![loss](http://i2.bvimg.com/604224/f74fb794fe134c49.png)

分为分类cls损失和坐标回归reg损失两部分

![xy](http://i4.bvimg.com/604224/1c88231b323dd70d.png)

其中`tx, ty, tw, th`四个量也就是output1中`40 * 60`的feature map上每个位置产生的`9`个anchors的每个`anchor`的四个量
`tx*, ty*, tw*, th*`是由label计算出来的坐标的偏移量

---

## RPN总结

因此综上所述RPN的数据流向是这样的：

`[n(3), n(3), p(512), 512]`的卷积核在作用在`[40, 60, p(512)]`的feature map上
得到`[40, 60, 512]`维的特征（也就是每个位置生成512维的特征），然后在得到的这个feature map
上的每个位置，用全连接层`[512, 54]`生成`[40, 60, 54]`的输出，然后切割成`[40, 60, 9, 4]`和`[40, 60, 9, 2]`的两个输出

其中`[40, 60, 9, 4]`代表了`40 * 60`的feature map上每个位置产生的`9`个anchors的每个`anchor`的四个坐标（的偏移量）

其中`[40, 60, 9, 2]`代表了`40 * 60`的feature map上每个位置产生的`9`个anchors的每个`anchor`的两个概率

---

# RPN与Fast-RCNN共享计算量
RPN接收feature map生成region proposal，Fast-RCNN接收feature map和region proposal并对region proposal进行分类和位置微调
两者都需要feature map，因此作者希望同一张图片计算feature map的计算量可以共享。因此设计了如下一个别有用心的训练过程

1. 首先单独训练RPN至收敛
2. 用第一步训练的RPN产生region proposal以此来训练一个Fast-RCNN
(到这一步两者都没有计算量上的共享)
3. 用第2步训练好的Fast-RCNN计算feature map的值初始化RPN这部分的参数，并对剩余部分也就是subnetworks进行fine-tuning
4. 保持计算feature map这段参数不变，用3中fine-tuning完的RPN产生region proposal给Fast-RCNN，fine-tuning其单独的参数
最终3、4两不停地交换直至收敛。

在inference阶段，base network计算出feature map，RPN从feature map中提取region proposal，Fast-RCNN对其进行分类和位置微调

---

参考资料

[How RPN (Region Proposal Networks) Works](https://www.youtube.com/watch?v=X3IlbjQs190)

[Faster-RCNN paper](https://arxiv.org/pdf/1506.01497)

申明：

本文仅是作者的个人学习经历总结而成，难免会有错误和遗漏，在此表示抱歉，望读者指正。