---
title: Linux下 Python2 与 Python3 版本切换
date: 2019-05-17 19:22:01
categories: 
- ["Linux"]
- ["语言","python"]
tags: 
- Python
- Linux
type: "tags"
---

> 说明：在Linux系统下，Python默认版本为2.7，但在使用过程中，可能经常需要使用Python3，因此，通过在网上的一系列搜索，找出了Python2与Python3的切换方法

# 一、添加软连接

```python
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 100
$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150
```

# 二、版本切换

> 输入如下命令后，在提示信息里面输入对应数字即可实现切换
>
> 添加过软连接后，以后需要切换python版本只用输入如下命令

```
$ sudo update-alternatives --config python
```

