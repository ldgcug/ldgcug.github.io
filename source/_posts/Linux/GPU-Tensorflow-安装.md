---
title: GPU Tensorflow 安装
date: 2019-05-21 11:32:35
categories: Linux
tags: 
- Tensorflow
- Linux
type: "tags"
---

> 此安装过程仅在Ubuntu16下安装测试

## 1、安装nvidia显卡驱动

```python
$ sudo add-apt-repository ppa:graphics-drivers/ppa
```

## 2、查看可安装的驱动版本

```python
$ ubuntu-drivers devices
```

![](GPU-Tensorflow-安装\drivers.png)

## 3、选择推荐版本号进行安装

```python
$ sudo apt-get install nvidia-390 nvidia-settings nvidia-prime
```

> 说明：我在安装的电脑上推荐的版本是390，因此使用的是390安装
>
> 但上图推荐的是430，在安装过程中需要更改为430，如：nvidia-430

```python
$ sudo apt-get install mesa-common-dev
$ sudo apt-get install freeglut3-dev
$ sudo reboot
```

## 4、安装cuda

> 从[cuda官网](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal)下载run文件，下载完后，执行下面代码命令

![](GPU-Tensorflow-安装\cuda.png)

```python
$ cd Downloads
$ sudo sh cuda_9.0.176_384.81_linux.run 
```

## 5、下载cudnn及安装cudnn

> [cudnn官网](https://developer.nvidia.com/cudnn)下载，下载7.5版本，对应CUDA9.0

> 百度云盘下载：
>
> 链接: https://pan.baidu.com/s/1cFoaZj_FRmQGXFneg9NL2A 提取码: 2a1v 

我这里是从云盘上下载，官网下载也差不多执行下面步骤

>下载完后解压，并进到该解压文件所在目录

```python
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

## 6、配置cuda环境

```python
$ sudo gedit ~/.bashrc
在最后添加如下两行代码
$ export PATH=/usr/local/cuda-9.0/bin:$PATH
$ export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
$ source ~/.bashrc
```

~/.bashrc相关的配置如下图

![](GPU-Tensorflow-安装\source.png)

## 7、安装gpu-tensorflow

```python
$ pip install tensorflow-gpu==1.6
```

> 说明：当安装1.6版本的tensorflow时，如果报错，可以尝试将版本降低，如改为 tensorflow-gpu==1.5 或 tensorflow-gpu==1.3

## 8、测试tensorflow安装成功与否

```
在python环境下输入import tensorflow，不报错，即安装成功
```

## 其他可能帮助信息

### cuda、cudnn版本对应关系

<https://www.tensorflow.org/install/source#tested_source_configurations>

![](GPU-Tensorflow-安装\cuda_cudnn.png)

### 查看GPU显卡型号和驱动版本

```python
$ lspci | grep -i nvidia
$ sudo dpkg --list | grep nvidia-*
网址：https://www.nvidia.cn/Download/driverResults.aspx/137427/cn
```

> 在服务器上安装时需要注意的事项

> 参考网址：<https://blog.csdn.net/QLULIBIN/article/details/78714596>
>
> （1）在安装cuda时，需要关闭图形化界面（ctrl+alt+F2键）
>
> （2）在出现的选项里，关于opengl的选择选择no，其他的选accept或yes。
>
> （3）如果在重启之后，可能在登陆界面一直循环往复，可能也需要到命令界面，切换为intel显卡