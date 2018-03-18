[RefineNet](https://arxiv.org/abs/1611.06612?context=cs.CV): Multi-Path Refinement Networks for High-Resolution Semantic Segmentation

全卷积网络（Fully Convolutional Network, FCN）用到语义分割（Semantic Segmentation）上有一个很大的问题的是经过多次的池化（pooling）
或者步长很大的卷积，特征图谱（feature map）的空间维度会变得特别小，包含了大量“是什么”的信息，而丢弃的大量“在哪儿”的信息，对于分类问题
来说“是什么”的信息就够用了，但是对于语义分割，“在哪儿”的信息同样很重要，这篇论文提出了使用大量中间层次的feature map来refine分割结果

**本文由本博主原创，版权归本博主所有，转载请注明出处，否则将依法追究责任**

---

# FCN使用在语义分割上的问题

论文强调：
卷机网络中不同阶段的池化和卷积的步长将图像的空间尺度降为了原来的1/32，因此丢失了细节的图像结构，带来局限，有三种方法来解决这个局限。

* 使用反卷积操作（deconvolution）把空间维度再提上去，但是反卷积操作并不能复原低层次的特征，因此无法输出准确的高分辨率的预测。
* [DeepLab](https://arxiv.org/abs/1606.00915)使用了空洞卷积（dilated convolution）的操作用来提升感受野同时不降低图像的尺寸，但是空洞卷积在高维度的特征上做卷积
导致需要大量的计算资源，会耗光GPU显存，因此一般只能输出原始图像1/8尺寸的预测
* 使用中间层的特征来产生高分辨率的预测，如FCN中的skip-connection一样。

在这边论文中，把skip-connection做的更为彻底，认为所有层次的特征对语义分割都有作用，高层次的特征决定了图像区域是什么，低层次的特征帮助产生更细节
的预测。因此作者提出了使用多尺度的特征来生成高分辨率的预测的模型。



# 模型

---

![image](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/refinenet/image.png)

该图展示了目前常见的语义分割模型以及RefineNet的总体框架

(a)图说的是常见的分类模型（如vgg16, ResNet）等等会把图像的尺寸缩小为原来的1/32
(b)图说的是使用空洞卷积的时图像的尺寸不下降会导致GPU显存爆炸。
(c)图就是RefineNet的具体内容了，下面具体来说一下

## RefineNet框架
作者希望使用多水平的特征太做高分辨率的预测。

作者将预训练的ResNet根据图像尺寸分成了四个块（block），应用ResNet-m标记第m个特征，并为每一个块配置了一个与之对应的RefineNet Block，
以RefineNet-m来标记对应ResNet-m的RefineNet Block。尽管这四个RefineNet Block的内部结构一样，但是里面的参数却是不共享的，使得模型更加的
灵活。从RefinNet-4只有一个输入也就是ResNet-4，在下一个阶段，RefineNet-4的输出和ResNet-3共同构成了RefineNet-3的两路输入，具体是用ResNet-3高分辨率的
特征来refine低层次的RefineNet-4的输出。RefineNet-2和RefineNet-1同样如此循环下去。这样做的目的是用融合不同阶段的特征。最终的输出
的特征做了一个softmax，然后用双线性插值恢复到原始图像的尺寸。

## RefineNet Block
一共有四个RefineNet Block，看一下内部的结构
![refinenet](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/refinenet/refinenet.png)

### Residual Convolution Unit（RCU）
除了RefineNet-4只有一路输入，其余的Block全部都有两路输入，每一路输入都要经过两个Residual Convolution Unit（RCU）来提取特征。
每一个RCU都是简化版的ResNet里的Block（见上图a），不同的是batch normalization被拿掉了。对于RefineNet-4卷积核的数量是512，对于
其余的RefineNet Block来说卷积核数量是256，这样产生的输入进入下一个子模块（融合层）进行融合。

### Multi-Resolution Fusion
对于不是RefineNet-4的Block，经过上一层RCU的处理后有两路尺寸不同的输入特征（一个是另一个的两倍），因此在这里的融合模块需要对不同
分辨率的特征进行融合。具体的做法是每一路的输出都先做一个三乘三的卷积作为一个输入的调整，然后把小的那个特征进行上采样（up-sampling）
变成大的特征的尺寸，最后两个出入的尺寸都相同了，进行求和并作为下一个子模块的输入。对于融合这个子模块，如果只有一路输入，如Refinet-4，
则该输入直接通过融合层，不做任何变化。

### Chained Residual Pooling
融合层的输入进到这层进行处理，具体的处理方式是先激活（relu）以下然后分为两路（见上图c），上一路不停地做池化卷积（步长为1，特征大小不变），一共做两次，
每一次都与下路进行求和得到输出喂给最后的Output Conv层。Chained Residual Pooling的主要作用的是从大的图像区域中抓取出背景的内容。

### Output Convolutions
上一层Chained Residual Pooling输出的特征还要经过这一层的一个RCU的处理，对于RefineNet-1也就是最后一个block，还放置了额外的两个RCU。

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