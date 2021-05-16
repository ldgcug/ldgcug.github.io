---
title: ubuntu16软件安装
date: 2019-05-21 16:40:39
categories: Linux
tags: Linux
type: "tags"
---

## 前言

> 仅用来在ubuntu上安装一些平时常用软件

## 1、Teamviewer安装

> [Teamviewer下载](https://www.teamviewer.com/cn/download/linux/)，Ubuntu系统下，文件默认下载到~/Downloads目录下

> 安装步骤

```python
$ cd ~/Downloads
$ sudo dpkg -i *.deb
```

> 在执行上面的sudo dpkg -i步骤后，会出现一个Error报错，不用着急，执行下面命令处理依赖即可

```python
$ sudo apt-get install –f 
```

> 说明：有时候会遇到`Teamviewer`无法打开，即双击`Teamviewer`无法显示，此时只需要命令启动即可

```python
$ teamviewer --daemon stop

$ teamviewer --daemon start
```

## 2、搜狗输入法安装

> [下载地址](https://pinyin.sogou.com/linux/)，Ubuntu系统下，文件默认下载到~/Downloads目录下

> 安装步骤 

```python
$ cd ~/Downloads
$ sudo dpkg -i *.deb
```

> 在执行上面的sudo dpkg -i步骤后，会出现一个Error报错，不用着急，执行下面命令处理依赖即可

```python
$ sudo apt-get install –f 
```

> 输入法配置

（1）安装完成后，在电脑设置里面找到Language Support

（2）键盘输入方式选择：fctix

![](ubuntu16软件安装\sogou.png)

（3）若没有fctix，在终端输入命令进行安装

```python
$ sudo apt-get install fcitx
```

（4）注销退出，重新登录进去

> 如果还不能切换中文输入法，参考该[网址](http://jingyan.baidu.com/article/adc815134f4b92f722bf7350.html)
>
> 若出现中文输入乱码情况，解决方法如下：

```python
$ cd ~/.config
$ rm -rf SogouPY* sogou*
```

> 执行完后，重启电脑即可

## 3、Vscode安装

```python
$ sudo add-apt-repository ppa:ubuntu-desktop/ubuntu-make
$ sudo apt-get update
$ sudo apt-get install ubuntu-make
$ sudo umake ide visual-studio-code
```

> 安装完成后，log out，然后在打开，就能在应用里看到Vscode

## 4、sublime text3安装

```python
$ sudo add-apt-repository ppa:webupd8team/sublime-text-3
$ sudo apt-get update
$ sudo apt-get install sublime-text-installer
```

## 5、pycharm命令行安装

> 添加源

```python
$ sudo add-apt-repository ppa:mystic-mirage/pycharm
```

> 安装免费社区版

```python
$ sudo apt update
$ sudo apt install pycharm-community
```

> 以前在u14上安装pycharm的另一个方法笔记（感觉还是上面一个好一点，上面是最新的笔记）

```
$ sudo add-apt-repository ppa:ubuntu-desktop/ubuntu-make
$ sudo apt-get update
$ sudo apt-get install ubuntu-make
$ umake ide pycharm
```

