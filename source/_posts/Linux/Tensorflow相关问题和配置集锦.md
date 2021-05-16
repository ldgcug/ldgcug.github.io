---
title: Tensorflow相关问题和配置集锦
date: 2019-05-30 23:49:12
categories: Linux
tags: 
- Tensorflow
- Linux
type: "tags"
---

## 前言

> 在做人工智能相关的训练方面，离不开Tensorflow，但是在安装或配置过程中都会遇到过一些问题，并且在做某一些特定的事也会有些小问题

## 一、配置方面

### 1、Tensorflow指定CPU训练

> 在机器上进行训练时，有时候可能出现多个python程序在训练的情况，并且其他人使用的是GPU训练，而这时我在使用GPU训练会报错误，此时，可以在程序中添加代码以指定CPU进行训练，添加代码如下：

```
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

参考网址：[网址1](https://blog.csdn.net/qq_35559420/article/details/81460912)

## 未完待续

