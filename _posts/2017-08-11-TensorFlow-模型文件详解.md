TensorFlow模型文件在模型训练完成后被保存（序列化）到本地硬盘用来做部署（deploy）、推断（inference）或者继续训练，理解这些文件的内容能够更好地帮助我们操作模型，本文会详细介绍保存到本地的模型文件结构和内容。

* *本文假定读者理解`TensorFlow`的图运算机制，如若不然，请参考此[教程](https://www.tensorflow.org/get_started/mnist/beginners)*

*若部分链接若打不开，请想一想为什么*


**本文由本博主原创，版权归本博主所有，转载请注明出处，否则将依法追究责任**

---
# 初探Protocal Buffer
`TensorFlow`所有模型文件都是以`protocal buffer`的形式保存，详细介绍参考[官网](https://developers.google.com/protocol-buffers/)。

这玩意是干嘛的？直接来看个[例子](https://developers.google.com/protocol-buffers/docs/pythontutorial)好了。
新建一个文件叫`addressbook.proto`，内容如下
```
syntax = "proto2";

package tutorial;

message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    required string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```
这是个通讯录的例子，通讯录里的基本单元是联系人，每个联系人有姓名`name`，邮箱`email`和电话号码`PhoneNumber`
等等属性(属性太多，就不一一说了)，然后编译这个文件(没有安装`Protocal Buffer`的读者就没必要自己动手实践了，这里只是个小例子
重要的是他的工作机制而不是例子本身)
```
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto
```
这样生成了`DST_DIR`路径下面生成了`addressbook_pb2.py`这个文件，这个文件包含一个类`Person`，这个类长什么样也不重要，重要的是他包含了定义在刚刚这个文件里
的这些属性(如(`name`, `email`等等))，来尝试用一下这个类
```
import addressbook_pb2
person = addressbook_pb2.Person()
person.id = 1234
person.name = "John Doe"
person.email = "jdoe@example.com"
phone = person.phones.add()
phone.number = "555-4321"
phone.type = addressbook_pb2.Person.HOME
```
这样就实例化了一个人`John Doe`以及附带的一些个人信息了。

所以到这里基本就能看清楚这玩意是干嘛的了：
>用户在文本文件中定义结构化的数据，`Protocal Buffer`由此生成各种语言都能加载、保存和使用数据的类

`TensorFlow`的模型文件就是这样首先定义在了文本文件中数据结构，然后由`Protocal Buffer`生成对应的类，使得任何语言(如`Python`,`Java`)都能
解析、加载和读取其中的数据。

----
# 保存模型生成模型文件

下面以计算$1 + 2 = 3$为例构建计算图并保存该模型(运算图)。如不能理解如下代码，请自行阅读官方教程

```Python
import tensorflow as tf

# build graph
a = tf.Variable(1)
b = tf.constant(2)
c = tf.add(a, b)

# launch session, run result
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("1 + 2 = %d" % sess.run(c))

# save model
saver = tf.train.Saver()
saver.save(sess, "<your/path/to/save/model>/model")
```
运行如上的程序后，model-file文件夹下会出现四个文件：
`model.meta`
`model.data-00000-of-00001`
`checkpoint`
`model.index`

其中`model.meta`称为元图(meta graph)，定义其结构的文本文件位于[此](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto)，
保存了恢复运算图所需要的全部东西，`model.data-00000-of-00001`保存了运算图中所有的变量(Variable)的值，有了这两个东西就能重新完整地恢复出以前的模型。

----
# 详解模型文件

在`TensorFlow`中任何以`def`结尾的变量或者类(如`graph_def`, `meta_graph_def`, `collection_def`)全部都是由`protocal buffer`生成的。

## 元图(meta graph)
前面我们说`model.meta`元图保存了恢复运算图所需要的全部东西，为什么这么说呢，看一下这个文件的具体内容，解析元图的代码如下
```
import tensorflow as tf

meta_graph_def = tf.MetaGraphDef()
with tf.gfile.FastGFile("model-file/model.meta") as f:
    model_file_string = f.read()
meta_graph_def.ParseFromString(model_file_string)
```
读者可以打印`meta_graph_def`就能详细地看到模型的内容了(内容比较大)，如果使用`dir(meta_graph_def)`则能查看`meta_graph_def`所有的属性和方法，最主要的几个属性如下：
`graph_def`：描述了运算图
`collection_def`：标记了运算中的一些特殊的元素，如可训练的变量`Variable`
`signature_def`: 标记了运算图中一些图书的节点，如输入是哪个张量，预测结果是哪个张量
`saver_def`：描述了保存器
`asset_file_def`：描述了模型所需要的额外的文件，如自然语言处理中预训练的词向量的文件是哪个
`meta_info_def`：描述了创建该运算图的`TensorFlow`版本等信息
下面来介绍其中比较重要的文件。

## 运算图的定义：graph_def
`graph_def`是元图`meta_graph_def`中最最重要的内容，没有`collection_def`等等其他的文件，`graph_def`也能够恢复原始模型至少9成以上的信息，原因在于她定义了完整的运算图，为什么这么说？看下面这个例子：

定义`graph_def`的文本文件在[此](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)，从这个文件的定义中就能
看出他所包含的属性`node`、`versions`和`library`，下面来具体看一下
```
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
```
到这里我们就已经获取了一个`graph_def`了，`dir(graph_def)`也确实能够看到这个`graph_def`
确实包含了`node`、`versions`和`library`这些属性。

`graph_def`定义了一个运算图，运算图的基本组件是节点(`node`)，`graph_def.node`会返回其包含的所有的运算节点组成的`list`

---

### 运算图的基本组件：node_def
`type(graph_def.node[0])`会告诉我们每一个`node_def`的类型是`tensorflow.core.framework.node_def_pb2.NodeDef`，
生成这个类的文本文件在[此](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto)

打印第一个节点看一下:
```
print(graph_def.node[0])
```
结果如下：
```
name: "Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_INT32
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_INT32
      tensor_shape {
      }
      int_val: 2
    }
  }
}
```
可以看出来，`a = tf.constant(2)`这行代码创造了这个节点，这个节点包含了如下内容：
这个节点的名字是`"Const"`
这个节点的操作是`"Const"`
这个常数的类型是`DT_INT32`
这个常数的值是`int_val:2`

打印第二个节点`print(graph_def.node[1])`会得到类似的内容，但是第二个节点的名字是`"Const_1"`而不是`"Const"`,而常数的值则变成了`int_val:3`，从这里这里发现 **`TensorFlow`中的节点的名字是坚决不允许重复的，原因在于会用节点的名字唯一地标记了一个节点！**

打印第三个节点`print(graph_def.node[2])`结果如下
```
name: "Add"
op: "Add"
input: "Const"
input: "Const_1"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}
```
这是一个加法`"Add"`节点,该节点有两个输入(`inputs`属性)(加法运算就应该有且只有两个输入)分别是`"Const"`和`"Const_1"`。

从定义`node_def`的文本文件中可以看出`node_def`包含如下的东西`op`、`name`、`input`、`attr`和`device`，总结一下
* `op`标记了这个节点的运算类型，如加法`Add`、减法`Sub`或者产生常数张量`Const`
* `name`标记了这个节点的名字，如`Add`，`Sub_1`，`variable_scope/Const_2`等，在一个运算图中每一个节点的名字唯一
* `input`标记了这个节点的输入是什么，如加法运算需要两个输入，可以是`["Const_1", "Const_2"]`
* `attr`标记了这个节点的一些其他属性，比如`Const`这个`op`的值是多少、`MatMul`矩阵乘这个`op`需不需要对对第二个矩阵转置等等,
其实`attr`也是一个[文本文件](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)生成的`Protocal Buffer`，里面的属性很多，不再一一叙述，有兴趣的读者可以直接查看源文件
* `device`标记了这个`op`所运行的设备，一般为空，同时重要性不大，不再叙述

---

### 运算图的版本：versions
`graph_def.versions`返回运算图的版本，关于运算图的版本，参考[官方教程](https://www.tensorflow.org/programmers_guide/data_versions)，因为不常用，不再叙述。

---

从以上内容我们就可以看出`graph_def`的工作原理了：

**`graph_def`包含了运算图中所有的运算节点`node`，每一个`node`包含了自己的信息包括这个节点的名字如`"Const_1"`，操作类型如`"Add"`,以及输入如`"Const"`和`"Const_1"`，每一个`node`各司其职，确保每一个张量的流动都是正确的，就能保证整个运算图的正确性了**

`graph_def`因为包含了完整的运算图而经常被单独序列化为文件，如官方提供的`inception-v3`的模型[下载](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)
我们也可以把我们这个$2 + 3 = 5$的运算图单独保存下来，执行如下操作即可：
```
graph_def_string = graph_def.SerializeToString()
with tf.gfile.FastGFile("two_plus_three.pb", "wb") as f:
    f.write(graph_def_string)
```
这样就把这个运算图保存为了`two_plus_three.pb`这个文件

下面演示怎么读取文件并运行运算图，代码如下：
```
import tensorflow as tf

graph_def = tf.GraphDef()
with tf.gfile.FastGFile("model-file/two_plus_three.pb", "rb") as f:
    graph_def_string = f.read()
graph_def.ParseFromString(graph_def_string)

tf.import_graph_def(graph_def, name="")

sess = tf.Session()
print(sess.run("Add:0"))
```
打印出的结果是$5$,也就是$2 + 3 = 5$

因为`protocal buffer`文件都是可以修改的，我们甚至可以做一些"非法"但是有趣的操作，比如把这个加法"强行"变成减法，演示如下：
```
import tensorflow as tf

graph_def = tf.GraphDef()
with tf.gfile.FastGFile("model-file/two_plus_three.pb", "rb") as f:
    graph_def_string = f.read()
graph_def.ParseFromString(graph_def_string)

graph_def.node[2].name = "Sub"
graph_def.node[2].op = "Sub"
tf.import_graph_def(graph_def, name="")

sess = tf.Session()
print(sess.run("Sub:0"))
```
得到结果是$-1$
其中
```
graph_def.node[2].name = "Sub"
graph_def.node[2].op = "Sub"
```
修改了以前加法运算的名称(`name`)和操作(`op`)，因此$2 + 3 = 5$被修改成了$2 - 3 = -1$

**此处仅为帮助读者理解`graph_def`的工作机制，不建议读者强行无意义地修改任何运算图！**

---

## 标记特殊元素：collection_def
我们说`collection_def`标记了运算图中的一些特殊元素
读取第2.节中保存的`meta_graph_def`并打印`collection_def`
```
import tensorflow as tf

meta_graph_def = tf.MetaGraphDef()
with tf.gfile.FastGFile("model-file/model-file/model.meta", "rb") as f:
    model_file_string = f.read()
meta_graph_def.ParseFromString(model_file_string)
print(meta_graph_def.collection_def)
```
结果如下
```
{'trainable_variables': bytes_list {
  value: "\n\nVariable:0\022\017Variable/Assign\032\017Variable/read:0"
}
, 'variables': bytes_list {
  value: "\n\nVariable:0\022\017Variable/Assign\032\017Variable/read:0"
}
}
```
这是在说这个运算图中的`variables`和`trainable_variables`分别是什么，按照[官方的说明](https://www.tensorflow.org/programmers_guide/meta_graph)
> CollectionDef map that further describes additional components of the model, such as Variables, tf.train.QueueRunner, etc.

意思是`collection_def`描述了模型中的额外的组件，如变量`Variables`和队列`QueueRunner`(队列曾经是`TensorFlow`官方推荐的一种输入方式，可惜快要被废弃(deprecated)了，详情参考这个[issue](https://github.com/tensorflow/tensorflow/issues/7951)，这里就不细说了)

`collection_def`因为保存了所有(可训练)的变量(的名字)，而在`meta_graph_def`中起了重要的左右，实际上从`TensorFlow`的加载模型的核心函数`tf.train.import_meta_graph`的源代码
```
# call `tf.import_graph_def`
importer.import_graph_def(
    input_graph_def, name=(import_scope or ""), input_map=input_map,
    producer_op_list=producer_op_list)

scope_to_prepend_to_names = "/".join(
    [part for part in [graph.get_name_scope(), import_scope] if part])

# Restores all the other collections.
for key, col_def in meta_graph_def.collection_def.items():
  # Don't add unbound_inputs to the new graph.
  if key == unbound_inputs_col_name:
    continue
  if not restore_collections_predicate(key):
    continue

  kind = col_def.WhichOneof("kind")
  if kind is None:
    logging.error("Cannot identify data type for collection %s. Skipping.",
                  key)
    continue
  from_proto = ops.get_from_proto_function(key)
  if from_proto and kind == "bytes_list":
    proto_type = ops.get_collection_proto_type(key)
    for value in col_def.bytes_list.value:
      proto = proto_type()
      proto.ParseFromString(value)
      graph.add_to_collection(
          key, from_proto(proto, import_scope=scope_to_prepend_to_names))
```

可以看出这个函数调用了`tf.import_graph_def`然后将`collection_def`中的所有元素全部恢复到`graph.collection`中，这样
`saver`才会从`collection`中找到需要恢复(restore)的变量，如果仅仅调用`tf.import_graph_def`这个函数，`saver`将会报出无法找到`Variable`可恢复的错误。

---

## 标记输入输出：signature_def
`signature_def`顾名思义签名，标记了运算图中某些特殊的张量，比如一个模型的输入经常是一个`tf.placeholder`，如果在构建图的时候不对这个张量加以命名(实际上也没多少人会主动命名)，这个
张量的名字就会是`Placeholder:0`(代表`Placeholder`这个`op`产生的第一个输出(实际上也只有一个))，输出可能是`Softmax:0`这个张量，这些名字都太变态了，不是`TensorFlow`的熟练使用者几乎不会
认出这个张量就是输入那个张量是输出，因此`signature_def`中如果保存了`{"inputs": "Placeholder:0"， "probability": "Softmax:0"}`这些信息，模型的使用者会更容易判别计算图的输入和输出

---

## 标记保存器saver：saver_def
`saver.restore`实际上也是一个`op`，`saver_def`记载了这个`op`的一些信息(如名字)，实际上不需要对`saver_def`有太多的了解，只需要知道``
`saver = tf.train.import_meta_graph()`生成的这个`saver`就是由`saver_def`所描述的就够了

---

# 总结

最后总结一下`TensorFlow`模型文件的结构和内容吧！


`meta_graph_def` (元图)
 * `graph_def` (运算图结构)
   * `node_def` (运算图节点)
     * `op` (节点运算类型)
     * `name` (节点名)
     * `input` (节点输入)
     * `attr` (节点特征)
   * `versions` (运算图版本)
 * `collection_def` (变量、队列等元素)
 * `signature_def` (输入输出张量)
 * `saver_def` (保存器)


PS

说`meta_graph_def`位于最顶层其实还不太正确，因为还有比她更"大"的数据结构，而且这个数据结构还紧密的跟服务`TensorFlow`模型的模块`TensorFlow Serving`联系在一起。

具体内容，请看《TensorFlow:SavedModel和TensorFlow Serving》一文。

---

内容部分参考

[Protocol Buffer Basics: Python](https://developers.google.com/protocol-buffers/docs/pythontutorial)

[A Tool Developer's Guide to TensorFlow Model Files](https://www.tensorflow.org/extend/tool_developers)

[TensorFlow Data Versioning: GraphDefs and Checkpoints](https://www.tensorflow.org/programmers_guide/data_versions)

申明：

本文仅是作者的个人学习经历总结而成，难免会有错误和遗漏，在此表示抱歉，望读者指正。