在`TensorFlow` 0.8版本时提供了分布式的支持，训练速度因此得到大幅提升，`TensorFlow`提供了一些工具使得多GPU的训练的代码能够跟单块CPU训练的代码差不多，本文将focus在多GPU的训练上，会依次介绍
* 多GPU训练的流程图
* 多GPU训练的代码实现

本文假定了读者
* *理解`TensorFlow`的图运算，如若不然，请参考此[教程](https://www.tensorflow.org/get_started/mnist/beginners)*
* *知道`TensorFlow`的求导运算，如若不然，请参考《TensorFlow：优化器与符号求导机制》一文*
* *知道机器学习的基本概念(如参数，损失函数等)，如若不然，请参考《统计学习方法》*

*若部分链接若打不开，请想一想为什么*

**本文由本博主原创，版权归本博主所有，转载请注明出处，否则将依法追究责任**

---

# 多GPU训练的流程图

多GPU训练能够让训练速度有非常大的提升，先来看一看具体的流程。

![multi-gpus](https://raw.githubusercontent.com/zakizhou/zakizhou.github.io/master/images/multi-gpus/mmulti-gpus.png)
这个流程的具体意思是有
* 变量创建在`CPU`上
* 每个`GPU`称为一个`tower`, 每个`GPU`上都会放置一份模型
* 每个`GPU`上的模型从`CPU`处拿到当前的变量(权值)，并单独处理属于自己的输入，得到损失，并计算损失相对于变量的导数
* 等待所有的`GPU`的导数都算完之后，`CPU`收集这些导数(算的快的`GPU`需等待慢的`GPU`，称为同步)
* 在`CPU`上对所有收集来的导数进行平均，并对变量(权值)进行更新

# 多GPU训练的代码解读
先来看一看单块CPU是如何执行训练的
```
import tensorflow as tf


inputs = ...
labels = ...
loss = calculate_loss(inputs, labels)

optimizer = tf.train.GradientDescentOptimizer(0.01)
grads = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads)
```
当然在单块`CPU`情况下后两行代码可以更简洁的写成`optimizer.minimize(loss)`
但是到了分布式不行，在1.中提到，需要收集每一个`GPU`计算出来的导数(放在一个`list`中)，并处理，也就是说每一个`GPU`都会生成一个`grads`的变量，其具体形式为(假设有三个变量):
`[("grad0", "var0"), ("grad1", "var1"), ("grad2", "var2")]`
在两块`GPU`中的形式为
`GPU:0`:`[("grad0_gpu0", "var0_gpu0"), ("grad1_gpu0", "var1_gpu0"), ("grad2_gpu0", "var2_gpu0")]`
`GPU:1`:`[("grad0_gpu1", "var0_gpu1"), ("grad1_gpu1", "var1_gpu1"), ("grad2_gpu1", "var2_gpu1")]`
放在一个`list`中的结果为
```
[[("grad0_gpu0", "var0_gpu0"), ("grad1_gpu0", "var1_gpu0"), ("grad2_gpu0", "var2_gpu0")],
[("grad0_gpu1", "var0_gpu1"), ("grad1_gpu1", "var1_gpu1"), ("grad2_gpu1", "var2_gpu1")]]
```
**注意`"var0_gpu0"`跟`"var0_gpu1"`是一个东西**

现在需要一个函数对这个收集到的梯度进行处理，于是就有了如下的[函数](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py)(选自官方教程`cifar10`的多`GPU`训练脚本)：
```
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #  ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
```
这个函数把
```
[[("grad0_gpu0", "var0_gpu0"), ("grad1_gpu0", "var1_gpu0"), ("grad2_gpu0", "var2_gpu0")],
[("grad0_gpu1", "var0_gpu1"), ("grad1_gpu1", "var1_gpu1"), ("grad2_gpu1", "var2_gpu1")]]
```
变成了`[("mean_grad_0", "var0"), ("mean_grad_1", "var1"), ("mean_grad2", "var2")]`
也就做到了收集梯度，并进行平均的功能，最后只需要`apply_gradients`就可以了，来看看完整的多`GPU`训练的[代码](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py)

```python

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr) # (1)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus): # (2)
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope, image_batch, label_batch) # (3)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables() # (4)

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss) # (5)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads) # (6)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads) # (6)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step) # (7)
```
一些解释：
* 首先在`CPU`上定义了优化器`opt`
* 对`GPU`进行循环，在每个`GPU`上算出`loss`，**reuse variable**，计算导数，并收集导数
* 在`CPU`上对收集好的导数进行平均，并进行参数的更新

---

参考链接

[Convolutional Neural Networks](https://www.tensorflow.org/tutorials/deep_cnn)

[cifar10_multi_gpu_train.py](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py)

申明：
本文仅是作者的个人学习经历总结而成，难免会有错误和遗漏，在此表示抱歉，望读者指正。