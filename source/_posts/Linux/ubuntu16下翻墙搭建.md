---
title: ubuntu16下翻墙搭建
date: 2019-05-18 08:47:13
categories: Linux
tags: Linux
type: "tags"
---

>  ### 说明：此教程为ubuntu下如何翻墙教程，相对来说步骤比较繁琐，可能有更简单的方法，以后用到会继续更新

> 需要安装软件：
>
> + chrome浏览器
> + shadowsocks-qt5

# 安装步骤：

## 1、ubuntu16安装chrome浏览器

```python
$ sudo wget http://www.linuxidc.com/files/repo/google-chrome.list -P /etc/apt/sources.list.d/
$ wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install google-chrome-stable
```

## 2、安装shadowsocks-qt5

```python
$ sudo add-apt-repository ppa:hzwhuang/ss-qt5
$ sudo apt-get update
$ sudo apt-get install shadowsocks-qt5
```

## 3、shadowscoks-qt5配置

> 打开`shadowsocks-qt5`

![](ubuntu16下翻墙搭建\add_manually.png)

> 在如下界面，设置IP、端口、密码等等
>
> 如果没有设置服务器IP，需要先搭建一个服务器，然后设置对应的端口及密码等等......（这里不介绍如何搭建翻墙服务器）

![](ubuntu16下翻墙搭建\ip_config.png)

## 4、添加插件

> 添加Proxy SwitchOmega.crx插件

插件下载地址：<https://github.com/FelisCatus/SwitchyOmega/releases>

说明：该插件下载地址还没有使用过，之前下载过该插件的其他地址，若这个不能使用，还是使用存在于硬盘的插件

![](ubuntu16下翻墙搭建\proxy_switch_download.png)

> 将该插件直接拖曳到chrome的扩展中，若拖曳失败，不能正常拖曳，参考该网址：<https://blog.csdn.net/qq_33033367/article/details/80952291>

注意：要切换为开发模式，然后进行拖曳，注意图中红色框

![](ubuntu16下翻墙搭建\proxy.png)

## 5、switchomega配置

![](ubuntu16下翻墙搭建\proxy_server.png)

>  下图中的URL地址需要自己手动输入，直接复制下面代码即可

```
https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt
```

![](ubuntu16下翻墙搭建\auto_switch.png)



> 最后点击Download ProfileNow

## 6、chrome浏览器访问google

> 首先需要打开shadowsocks-qt5，然后连接服务器

> 然后通过切换规则，切换为auto switch，就可以访问google了

![](ubuntu16下翻墙搭建\google.png)

> 打开浏览器，输入[www.google.com](http://www.google.com)，如果出现这种问题，修改为proxy即可

![](ubuntu16下翻墙搭建\add_condition.png)