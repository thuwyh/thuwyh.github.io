---
categories:
- Review
date: "2017-03-20T00:00:00Z"
tags:
- tensorflow
- 编译
- 树莓派
title: 用makefile编译tensorflow
---
官方指南见
[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile)

前几天已经用bazel编译了一遍tensorflow，但是如果想在嵌入式平台例如树莓派上使用这个框架这条路可能行不通。因为受限于平台资源，可能无法用Bazel来编译（github已经有人成功了）。但已经有人提供了用makefile来进行编译的方案，可以编译出一个不含python绑定和gpu支持的静态库，非常适合在嵌入式平台使用。目前可用的目标平台有

- iOS
- OS X (macOS)
- Android
- Raspberry-PI

## 准备工作

- clone tensorflow repo到本地
- 以下**所有**的命令都应在在仓库的**根目录**下执行。首先执行`tensorflow/contrib/makefile/download_dependencies.sh`下载所需的依赖项。文件保存在`tensorflow/contrib/makefile/downloads/`目录下。如果是编译Linux版本，这步可以不执行，原因后面会提到。

## 编译Linux版本
安装必要的包`sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev git python`

执行编译，运行`tensorflow/contrib/makefile/build_all_linux.sh`。整个编译的过程用了一小会儿。README里说需要先执行`download_dependencies.sh`，其实在这个脚本里会清空downloads文件夹并重新下载一遍。。

验证。执行以下命令下载inception模型

```
mkdir -p ~/graphs
curl -o ~/graphs/inception.zip \
 https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip \
 && unzip ~/graphs/inception.zip -d ~/graphs/inception
```

再执行以下命令，**注意graph=后面有引号！**

```
 tensorflow/contrib/makefile/gen/bin/benchmark \
 --graph="~/graphs/inception/tensorflow_inception_graph.pb"
```

应该就成功了，如果提示找不到网络的话自己检查一下路径。

## 在树莓派上编译

```
tensorflow/contrib/makefile/download_dependencies.sh #跟之前一样，下载依赖库
sudo apt-get install -y autoconf automake libtool gcc-4.8 g++-4.8  #安装编译工具
```

以下编译protobuffer
```
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure
make CXX=g++-4.8
sudo make install
sudo ldconfig  # refresh shared library cache，很重要
```

然后可以验证一下
```
protoc --version
```
如果看到版本号应该就ok了。

然后编译Tensorflow，在编译选项里打开了针对树莓派平台的优化。
```
make -f tensorflow/contrib/makefile/Makefile TARGET=PI \
 OPTFLAGS="-Os -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize" CXX=g++-4.8
```
指定g++为4.8版本是为了避免Raspbian自带4.9编译时报错。前面的protobuffer我也加了指定，避免版本不一致产生问题。

经过漫长的等待，应该就能编译成功了，可以试下``tensorflow/contrib/makefile/gen/bin``目录下的``benchmark``程序。

我原先树莓派上装的是Ubuntu Mate，死活编译不成功，提示为找不到Probobuffer的函数，感觉是Makefile哪里出了问题，路径没搞对，换成Raspbian就一切顺利了。感觉可以学一波Makefile编写，然后检查下两个系统生成的文件到底有什么不同。
