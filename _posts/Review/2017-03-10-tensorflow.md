---
title: 编译安装Tensorflow
categories: 
    - Review
tags: 
    - Tensorflow
    - Deeplearning
---

如果采用官方包安装Tensorflow的话，有许多针对平台的优化都没有打开，会导致性能下降。因此，最佳的安装Tensorflow的方法是从源码编译。整个过程大概如下。

## 安装Bazel
### 安装JDK8
```shell
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer
```
下载安装包的时候速度极慢，打开VPN快了许多。。
### 安装其他依赖项
```
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
```
我还修改了系统的默认python到python3版本
##配置安装Tensorflow
下载Tensorflow的git仓库
```
git clone https://github.com/tensorflow/tensorflow 
```
进到目录里，checkout正确的版本
```
git checkout r1.0
```
生成编译配置
```
$ cd tensorflow  # cd to the top-level directory created
$ ./configure
```
编译pip文件
```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```
这步执行完后会生成一个脚本，用它可以生成whl包。执行
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```
生成包，包存在``/tmp/tensorflow_pkg``里，包名是自动的。然后再用pip安装这个包
```
sudo pip install /tmp/tensorflow_pkg/***.whl
```
其中***就是生成的包名，到刚才的目录下看一下就知道了。

##测试

既然tensorflow可以直接用pip方便地安装，为什么要千辛万苦编译呢？因为官方编译好的软件包为了提高通用性基本不可能发挥硬件的全部性能，很多优化开关都没有打开。而我们自己编译的版本则会根据硬件进行优化。为了测试性能差别，用以下简单的卷积网络在MNIST数据集上进行实验。

```python
import tensorflow as tf
import time
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
sess = tf.InteractiveSession()
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
t1=time.time()
t2=t1
for i in range(5000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    t0=t2
    t2=time.time()
    print("step %d, training accuracy %g, time %g"%(i, train_accuracy,t2-t0))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
t3=time.time()
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
t2=time.time()
print("test time %g"%(t2-t3))
```

测试结果是同样跑100个batch，pip安装的版本耗时21秒左右，而编译版本基本为15秒。提升还是比较明显的。