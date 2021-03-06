在执行随机梯度下降(Stochastic Gradient Descent)的过程中，需要损失函数(loss function)相对于所有可训练参数(trainable parameters)的导数并进行参数的更新，这个工作是由`tf.train.Optimizer`(子)类完成的。

本文假定了读者
* *理解`TensorFlow`的图运算和基本的机器学习知识(如损失函数)，如若不然，请参考此[教程](https://www.tensorflow.org/get_started/mnist/beginners)*

*部分链接打不开是由于GFW*

**本文由本博主原创，版权归本博主所有，转载请注明出处，否则将依法追究责任**

---



# 参数更新(或者叫训练)的例子和流程

以下的例子将优化$y = x^2$这个简单的函数
```
import tensorflow as tf

x = tf.Variable(3.)
y = tf.square(x)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        _, y_value, x_value = sess.run([train, y, x])
        print("step %d: x %f, y %f" % (step, x_value, y_value))
```
总结一下在`TensorFlow`中进行参数更新的一般步骤是
1. 定义目标(如损失函数) `y = tf.square(x)`
2. 初始化(某种)优化器(`optimizer`)
3. 定义训练或参数更新步骤(train op) `train = optimizer.minimize(y)`
4. 循环地执行训练步骤 `sess.run`

那么这个`train = optimizer.minimize(y)` 又是怎么做到极小化目标的呢，来看看源代码。

如下这段话来自`tf.train.Optimizer`的源代码
```
### Processing gradients before applying them.

Calling `minimize()` takes care of both computing the gradients and
applying them to the variables.  If you want to process the gradients
before applying them you can instead use the optimizer in three steps:

1.  Compute the gradients with `compute_gradients()`.
2.  Process the gradients as you wish.
3.  Apply the processed gradients with `apply_gradients()`.
```
这段话简要地说明了进行参数更新的全部过程，
1.用`optimizer.computer_gradients`计算出导数
2.对得到的导数进行处理(如根据动量`moment`)进行调整
3.使用`optimizer.apply_gradients`进行参数更新
而`optimizer.minimize`这个方法包含了以上的全部过程，隐藏了计算导数(梯度)的过程。

值得一提的是，`tf.train.Optimizer`是基类，已经实现了`compute_gradients`和`apply_gradients`两个方法，所有继承她的优化器(如`tf.train.AdamOptimizer`)将自动拥有这些方法，而不同的优化器之间的差别就在于第2步如何处理得到的导数，不同的优化器有不同的处理方法。

---

# 符号导数的计算
第一节中提到了`optimizer.compute_gradients`能够计算**符号导数**，下面来看一下她是如何计算出来的

![compute_gradients](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/gradients/compute_gradients.JPG)

这段代码是`tf.train.Optimizer`的`compute_gradients`方法的源代码，从中可以看出，除去一些条件的判断和处理，最核心的一步就是用`gradients.gradients`算出的导数，这个函数已经暴露出来，也就是`tf.gradients`函数，为了清晰地看清楚这个函数是怎么运行的，使用`TensorBoard`对运算图进行可视化。

---

## 极小化$y = \sin(x)$
```
import tensorflow as tf

x = tf.Variable(3.)
y = tf.sin(x)

gradients = tf.gradients(y, [x])

writer = tf.summary.FileWriter("<your/path/to/save/graph>")
writer.add_graph(graph=tf.get_default_graph())
```
最后两行代码保存了运算图的结构，使用`TensorBoard`可视化运算图，在shell(linux)或cmd(windows)里面执行
```
tensorboard --logdir <your/path/to/save/graph>
```
然后按照提示打开浏览器，点击`GRAPHS`一栏，就能看到如下结果

![sin(x)](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/gradients/example1.png)

其中`Variable`就是我们定义的`x`,然后与`Sin`节点连接，也就是执行的`y = tf.sin(x)`，而`gradients = tf.gradients(y, [x])`则添加了算导数的这些节点，$y = \sin(x)$的导数是$\frac{dy}{dx} = \cos(x)$，而在这个图当中`Variable`(`x`)也与`Cos`节点相连接(`mul`节点的另一个输入是$1$不影响结果)得到$\cos(x)$，这与我们手动算出的导数是一致的。

---

## 极小化$y = e^{\sin(x)}$
$y =sin(x)$的例子太简单了，来看一个稍微复杂的例子，极小化$y = e^{\sin(x)}$：
```
import tensorflow as tf

x = tf.Variable(3.)
y = tf.exp(tf.sin(x))

gradients = tf.gradients(y, [x])

writer = tf.summary.FileWriter("gradients/summary")
writer.add_graph(graph=tf.get_default_graph())
```
 运算图结构如下：

![exp](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/gradients/example2.png)

来看一下具体过程。首先创建`Variable`(`x`)，然后依次流经`Sin`节点和`Exp`节点得到$e^{\sin(x)}$，而$y = e^{\sin(x)}$的导数是
$$\frac{dy}{dx} = e^{\sin(x)}\cos(x)$$
在运算图中，`Exp`节点传入`Exp_grad`没有任何变化，而`Variable`又传到了`Cos`节点，在`Sin_grad`中的`mul`节点，`Exp`与`Cos`乘在了一起，也就是$e^{\sin(x)}\cos(x)$，导数由此就算出来了！

---

## 泛化到其他复杂函数
在2.1和2.2，我们分别查看了$y = \sin(x)$和$y = e^{\sin(x)}$的导数的计算过程，但是往往需要优化的一个函数(比如20层的前馈神经网络)太复杂了，不是这样小的函数，那怎么办呢？**锁链规则(chain rule)**(不知道锁链规则的请自行查阅任何一本名字中包含"高等数学"或者"数学分析"的书)将帮助我们解决这个问题。

定义$f_1(x) = sin(x)$,$f_2(x) = exp(x)$，那么第一个例子就变成了$y = f_1(x)$导数是$y = f_1'(x)$，第二个例子变成了$y = f_2(f_1(x))$导数是$y = f_2'(f_1(x))f_1'(x)$，实际上

**任何一个无论多么复杂的函数都可以写成$f_n(f_{n-1}...f_2(f_1(x)))$的形式**(有兴趣的读者可以自行证明这个结论)

有了这个结论就能明白`TensorFlow`的符号导数的机制了，无论多么复杂的导数，无非就是先算$f_n$的导数，再算$f_{n-1}$的导数以此类推最后算$f_1$的导数然后乘在一起即可，所以`TensorFlow`的符号导数的机制是：

**在`TensorFlow`中这些基础的函数$f_i(i=1,...,n)$(或者叫`op`，如`tf.exp`,`tf.sin`,`tf.log`)等等的导数是在实现这个`op`时就已经定义好的，用户使用这些的`op`构建目标（如损失函数），`TensorFlow`记住张量`Tensor`的运算的流程(`Flow`)最后再拼接这些`op`的导数计算出最终的导数**

---

## 源码解读
第一小节使用了`tensorboard`来对`tf.gradients`得到的运算图进行了可视化方便读者理解，有了以上的例子之后再来看`tf.gradients`是如何
做到自动求导的。

在`tf.gradients`的源码中定位到最重要的一步，也就是

![gradients](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/gradients/gradients.JPG)

`ops.get_gradient_function(op)`是在尝试找到`op`对应的导数，继续查看`ops.get_gradient_function`
这个函数的源代码，如下图所示

![get_gradient_function](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/gradients/get_gradient_function.JPG)

`_gradient_registry.lookup(op_type)`是核心代码，是从`_gradient_registry`这个东西中查到`op`，接续查看`_gradient_registry`的源代码

![gradient_registry](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/gradients/gradient_registry.JPG)

定义在`tensorflow\python\framework\ops.py`中，是一个`registry.Registry`的实例，在这个定义的下方一点有一个
`RegisterGradient`的类型定义

![RegisterGradient](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/gradients/RegisterGradient.JPG)

这是一个用来做装饰器的类型（因为`__call__`方法被定义了），可以看出就是这个函数将任意一个`Op`的导数链接给了这个`Op`，具体查看一下
`tensorflow\python\ops\math_grad.py`这个文件中的`tf.square`这个`Op`的导数

![square_grad](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/gradients/square_grad.JPG)

平方的导数是其自身的两倍，这个定义正是如此！

到此`tensorflow`的求导的源代码的核心就解读完了，一句话概括就是每一个`Op`都事先定义好了自己的导数，并使用`_gradient_registery`链接
到这个`Op`，在使用`tf.gradients`进行求导时，`tensorflow`会查看这个链接，并使用锁链规则将每个`Op`的导数结合在一起，得到最终每个`tf.Variable`
的导数。

---

## **自定义导数
`TensorFlow`是如何定义`op`的导数的呢，需要去查看源代码，在[自定义`op`](https://www.tensorflow.org/extend/adding_an_op)
这个教程中详细地介绍了每一个`op`的导数是如何计算出来的(需要看得懂c++的代码)

---

# 分布式中(导数)梯度的处理
在1.节中提到`tf.train.Optimizer`的`minimize`方法能够执行算梯度然后更新参数的一系列工作，实际上这对于单机的程序来说已经足够了，读者不需要去手动地依次执行算梯度，再处理梯度，最后更新参数的过程，但在分布式的情况下将完全不同，必须要手动处理这些梯度，为什么要手动？具体怎么处理？请看`TensorFlow：多GPU训练`一文！

---

参考链接：

[tensorflow的函数自动求导是如何实现的？](https://www.zhihu.com/question/54554389?from=profile_question_card)

[tf.gradients](https://www.tensorflow.org/api_docs/python/tf/gradients)

[Chain rule](https://en.wikipedia.org/wiki/Chain_rule)

申明：

本文仅是作者的个人学习经历总结而成，难免会有错误和遗漏，在此表示抱歉，望读者指正。