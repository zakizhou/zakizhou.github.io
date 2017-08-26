[Feature Pyramid Network](https://arxiv.org/abs/1612.03144) for Object Detection

Faster-RCNN中的RPN和Fast-RCNN都仅仅使用了使用了一个feature map（vgg16的`conv4_3`）作为输入，而Object Detection或者Semantic Segmentation
这些需要“在哪里”而不仅仅是“是什么”信息的任务，仅仅一个feature map会导致low level的“在哪里”的信息丢失，因此本文提出了Feature Pyramid Network
使得CNN网络中的多尺度的特征全都可以被用上来进行更高精度的预测。

**本文由本博主原创，版权归本博主所有，转载请注明出处，否则将依法追究责任**

---

# Object Detection中的一些困难

![FPN](http://thumbnail0.baidupcs.com/thumbnail/e75c6f5d3d6b1a8d3730867593df7538?fid=2419221423-250528-853011733514779&time=1503748800&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-IbyCmTLRfq4dHL12SCRtrIJP6zo%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=5516685682978040996&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)

论文指出：
识别尺寸不同的物体是计算机视觉中一个基础的挑战，基于[图像金字塔](http://docs.opencv.org/2.4/doc/tutorials/imgproc/pyramids/pyramids.html)(上图a所示)
的特征金字塔是解决这个挑战的一个方法。但是图像特征金字塔有非常大的局限性，在inference阶段，同样的图片的不同尺度被送入分类器（如CNN）会导致计算量成倍地增长，
而且希望端到端地训练这样的分类器也是很难的，因为显存不够，因为这些原因，Fast-RCNN和Faster-RCNN都没有使用这个方式而是使用了单个feature map。

但是图像金字塔并不是唯一的一种生成多尺度特征的方法，一个典型的卷积神经网络天然地拥有一个多尺度的特征图谱，这种图像内部的特征图谱
导致了特征尺度的差别，高分辨率的特征有low level的内容，降低了他们对图像识别的表现能力。

[Single Shot Detector](https://arxiv.org/abs/1512.02325)(ssd)是第一个使用卷机网络的金字塔状的特征来进行object detection的，
[U-Net](https://arxiv.org/abs/1505.04597)也使用了skip-connection来连接不同尺度的特征

这篇论文的目的就是使用卷积网络天然地金字塔状的特征，也就是这里的Feature Pyramid Networks。


# Feature Pyramid Networks
Feature Pyramid Networks使用单张图像作为输入，使用全卷积地方式，输出成比例多尺度的feature map。这个过程是独立于backbone的网络结构的，因为任何
一个卷积网络都有这样的形式，在这篇论文中作者使用了ResNet（又是她！），这个特征金字塔的构建包含了底到顶、顶到底和侧向的连接三个部分。
结果如下所示：
![image](http://thumbnail0.baidupcs.com/thumbnail/64264e1e14076ef530d6548390aed565?fid=2419221423-250528-217803025057677&time=1503738000&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-5HDN0fI1%2BDUVWh6u0MmnI4iRLow%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=5513686513751886572&dp-callid=0&size=c710_u400&quality=100&vuk=-&ft=video)

## 底到顶（Bottom-up pathway）

底到顶的这一路是一个卷积网络的前馈的计算过程。在一个卷积网络中有很多层会输出同样尺寸的特征，作者在这里将这些层整体看为一个阶段（stage），
作者选择每个阶段的最后一层的输出作为特征的参考集（也就是上图中图像的上方这些蓝色的框框，有三个阶段）。

具体地，作者使用了ResNets中每一个residual block的激活作为特征的参考集，记这些特征（`conv2`, `conv3`, `conv4`, `conv5`）分别为${C_2, C_3, C_4, C_5}$，他们对应的步长分别是${4, 8, 16, 32}$，
作者丢弃了`conv1`因为维度太大了，占用显存。

## 顶到底和侧向连接 （Top-down pathway and lateral connections）
顶到底的这一路使用上采样了的空间上模糊但是语义上丰富的特征来强化高分辨率低语义的特征，具体的强化方式就是侧向的连接。

上图就是顶到底的feature map的构建方式。首先将分辨率模糊但是语义丰富的特征进行上采样，使之维度变大2（使用的是最紧邻）。上采样完之后的特征与一个经过
一乘一的卷积（这里做卷积是为了保证两者特征的channels的相等）的对应的特征相加在一起，然后不停地继续做下去产生最终的特征图谱。为了开始
这个循环，先将$C_5$进行一乘一的卷积产生最粗糙的特征。对每一个加和得到的特征还要进行一个三乘三的卷积。最终的feature maps标记为${P_2, P_3, P_4, P_5}$，
对应于${C_2, C_3, C_4, C_5}$。


## 应用Feature Pyramid Networks
作者将Feature Pyramid Networks用在了Faster-RCNN中的两个阶段RPN和Fast-RCNN上，前者用来生成region proposal，后者用来做object detection。

## Feature Pyramid Networks for RPN
在原始的Faster-RCNN的paper中，RPN使用了一个三乘三的小窗口来扫feature map，生成一个固定长度（如512）的intermediate的特征，再将这个特征映射到两个
相邻的输出上，一个做cls判断anchor是否包含object，一个做reg对anchor进行位置的精修（作者称之为head），这是RPN的思路。将FPN应用在RPN上是很直接的，
之前RPN作用在一个feature map上，在这里只需要将RPN作用在FPN输出的各个尺度的特征上就行。但是如果每个feature map上都是用多个尺度多个纵横比的anchor
会导致这些anchor有很大的重叠，一个典型的判断就是维度小的feature map应该用来定位大的物体，维度大的feature map应该用来定位小的物体，因此作者照这个思路
设计了anchor的摆放。

正式地，定义${P_2, P_3, P_4, P_5, P_6}$对应的anchor的尺寸分别为${32^2, 64^2, 128^2, 256^2, 512^2}$，和原始paper一个，每一个尺寸的anchor都有三种
纵横比${1:2, 2:1, 1:1}$，这样五个feature map每个对应一个尺寸，每个尺寸有三个纵横比，一共就有十五种anchor。（从这里就可以看出$P_6$的维度是最小的，
其对应的anchor也最大，这表明了作者希望这个维度最小的feature map用来定位最大的物体）

标记正负anchor的方式与原始paper所一致。

需要注意的是，不同水平的feature map对应的head的参数是共享的，作者试验了不等的情况下的performance发现跟参数共享差不多，因此这表明了所有尺度的feature map
都有相似的语义水平。

## Feature Pyramid Networks for Fast-RCNN


---


# 个人总结
* 把**多尺度的特征**的融合做的很极致，也是后来给Feature Pyramid Network的提出带来提示的一篇论文。
* 再一次证明了low level的特征对于处局部图像的任务如Object Detection和Semantic Segmentation是有非常大的帮助的。

---

参考资料

[RefineNet](https://arxiv.org/abs/1611.06612)

[RefineNet实现](https://github.com/guosheng/refinenet)

[Feature Pyramid Network](https://arxiv.org/abs/1612.03144)


申明：

本文仅是作者的个人学习经历总结而成，难免会有错误和遗漏，在此表示抱歉，望读者指正。