---
title: (转)双系统之Win重装后linux引导损坏
date: 2019-07-01 20:37:27
categories: Linux
tags: 
- 系统
- Linux
type: "tags"
---

## 前言

> 由于一些需求，有时需要重装电脑，而本来是双系统的电脑，重装后，Linux的引导修复可能缺失，即在启动的过程中只有windows的界面，而没有win与linux的选择界面，因此，在网上找了几篇博客，几经对比后，还是这一片写的比较好（对我而言），并且已经测试成功

## 原文网址

[原文](https://blog.csdn.net/zhuoyinping7159/article/details/80546977)

## 安装步骤

### （1）首先，得制作一个ubuntu系统启动盘（不论14、16、18这样），制作方法这里不做说明

### （2）开机，从U盘进入（不同的电脑按键不一样，大部分是F12），选择try install ubuntu，不要选择安装ubuntu

### （3）进到ubuntu试用版后，连接wifi，打开终端（ctrl+alt+t）

### （4）在终端输入如下命令

- 添加源，更新

  ```
  sudo add-apt-repository ppa:yannubuntu/boot-repair
  sudo apt-get update
  ```

- 安装boot-repair

  ```
  sudo apt-get install -y boot-repair
  ```

- 安装成功后，在终端输入`boot-repair`，启动工具，在弹出的界面框中选择`recommended repair`，稍等一会，进行修复

- 修复完成后，重启电脑，此时重启过程中就能看到ubuntu的选项了

  我这里的选择项有点奇怪，但是暂时也不去管它了，倒数第二个是windows的启动项

  ![](转-双系统之Win重装后linux引导损坏/1.jpg)

## 总结

> 我按照上面的方法就完成了linux 的grub引导修复，在修复过程中没有出现原文中所遇到的windows引导损坏的问题，但还是在这里记录一下windows引导损坏的解决方法，如下

在linux的终端窗口中，输入

```
sudo update-grub
```

然后重启，就会出现双系统的选择界面了，这个还没有尝试，不过应该问题不大