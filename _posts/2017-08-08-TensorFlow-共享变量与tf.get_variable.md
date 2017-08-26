`TensorFlow`中一共有两种定义变量`Variable`的方式，分别是`tf.Variable`和`tf.get_variable`，初学者可能不知道后者或者以为两个功能一样，实际上后者拥有前者的全部功能，因为共享变量的机制而威力大增。

本文假定了读者
* *理解`TensorFlow`的图运算，如若不然，请参考此[教程](https://www.tensorflow.org/get_started/mnist/beginners)*

*若部分链接若打不开，请想一想为什么*

**本文由本博主原创，版权归本博主所有，转载请注明出处，否则将依法追究责任**

---

# 共享变量(sharing variables)的工作机制
想象一下要优化如下的函数：

$$\min_{x}f(x) = {\exp(\sin(4\cos(x) + 1))} + {\exp(\sin(5\cos(x) + 2))}$$

当然可以直接用`TensorFlow`的基础函数`op`写出这个目标函数，但是写出来保证非常丑陋，可读性太差，有没有更又没的办法呢？细心的读者会发现，加的两项是具有相同的结构:
$$\exp({\sin(a\cos(x) + b)})$$
将这个公式定义成一个使用$a, b$作为参数的函数，然后相加是不是显得很优雅？实现一下！
```Python
import tensorflow as tf


def func(a, b):
    x = tf.Variable(tf.random_uniform([]), name="x")
    mul = a * tf.cos(x)
    add = mul + b
    sin = tf.sin(add)
    exp = tf.exp(sin)
    return exp
```
其中先定义了一个待优化(可训练)的标量`x`然后算出对应的函数值，最后的目标变量就可以写成
```
object_func = func(4, 1) + func(5, 2)
```
这样就可以完工了？！**实际上并不是，为什么？**
```
graph = tf.get_default_graph()
print(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
```
打印出的结果是
```
[<tf.Variable 'x:0' shape=() dtype=float32_ref>,
<tf.Variable 'x_1:0' shape=() dtype=float32_ref>]
```
而每一个被创建的可优化的变量都会加入到`tf.GraphKeys.GLOBAL_VARIABLES`这个`collection`中，这个里面居然有两个变量，而实际上应该只有一个变量才对！

**错误的原因是每执行依次`func`这个函数，`x = tf.Variable(tf.random_uniform([]), name="x")`这行代码都会新建一个新的可优化的变量**

那么怎么样才能"共享"第一次创建的变量呢，共享变量(sharing variable)就是为这个而生的，看如下解决这个问题的代码：
```
import tensorflow as tf


def func(a, b):
    # change tf.Variable to tf.get_variable
    x = tf.get_variable(initializer=tf.random_normal_initializer(), shape=[], name="x")
    mul = a * tf.cos(x)
    add = mul + b
    sin = tf.sin(add)
    exp = tf.exp(sin)
    return exp

with tf.variable_scope("scope") as scope:
    add_left = func(4, 1)
    scope.reuse_variables()
    add_right = func(5, 2)
object_func = add_left + add_right
```
从这段代码可以看出
**主要的思想是将定义需要重用的变量的函数放在一个变量域`tf.variable_scope`下，然后定义好之后，每一次重用这个变量，就调用一次`scope.reuse_variables()`即可。**

---

# TensorFlow中共享变量的使用场景
可能有读者会说这个例子很极端，深度学习中这样的例子很少，实则不然，下面举两个用到共享变量的例子。

---

## 递归神经网络(recurrent neural networks)

*不知道递归神经网络是什么的读者请查看此[教程](https://colah.github.io/posts/2015-08-Understanding-LSTMs)*

第一个用到共享变量的地方是大名鼎鼎的递归神经网络，递归神经网络因为其权值是共享的，每向前传播一步，就要调用一次权值，因此在这里使用了共享变量，来看`TensorFlow`中递归神经网络的函数`tf.nn.static_rnn`的源代码

![rnn](http://i4.bvimg.com/604224/03d673595581232a.jpg)

为了减小篇幅，我对源代码进行了折叠只保留了跟本文有关的部分，从上面的代码可以看到，`time`就是递归网络向前传播的步数，第一次传播时会创建权值，之后调用这个权值需要`reuse`一下，这也就是
```
 if time > 0:
      varscope.reuse_variables()
```
所做的工作

---

## 验证集(validation set)
第二个用到共享变量的地方是验证集。一般的建模流程是这样的
```
def calcualate_logits(inputs):
    # define trainable variables
    # forward to calculate logits
    return logits
```
如果`inputs`是训练集的，根据`logits`则会计算训练集的损失，如果`inputs`对应验证集，则可以计算验证集的损失。因此这个函数需要计算两次
```
with tf.variable_scope("model") as scope:
    train_logits = calculate_logits(train_inputs)
    scope.reuse_variables()
    validate_logits = calculate_logits(validate_inputs)
```
训练集和验证集的损失都有了，才能画出这样一条深度学习中经常看见的曲线：

![curve](http://i2.bvimg.com/604224/c80843482710d721.jpg)

---

## 多GPU训练
第三个常用的地方在于多GPU训练，请参考`TensorFlow：多GPU训练`一文

---

参考链接

[Sharing Variables](https://www.tensorflow.org/programmers_guide/variable_scope)

申明：
本文仅是作者的个人学习经历总结而成，难免会有错误和遗漏，在此表示抱歉，望读者指正。


