---
title: 卸载原版gazebo并安装新版gazebo
date: 2019-05-22 15:25:46
categories: ROS
tags: ROS
type: "tags"
---

## 前言

> 从最初始在ubuntu14上安装ros indigo版本，到后面在ubuntu16安装ros kinetic版本，中间遇到过需要安装新版本gazebo的问题，如u14上安装ros后，默认安装gazebo2，可能需要改成gazebo7；u16上安装ros后，默认安装gazebo7，可能需要改成gazebo8

>  卸载`gazebo2.2`安装`gazebo7`网址：点击[此处](https://blog.csdn.net/tust123qht/article/details/78796617)

> 卸载`gazebo7`安装`gazebo8`步骤如下：

（1）卸载ros-kinetic-desktop-full

```python
$ sudo apt-get remove ros-kinetic-desktop-full
```

（2）卸载gazebo7

```python
$ sudo apt-get remove gazebo-* 
```

（3）安装gazebo8

```python
$ sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
$ wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install gazebo8
$ sudo apt-get install libgazebo8-dev
```

> 说明：第一步就卸载了ros-kinetic-desktop-full，因此需要重新安装ros-kinetic-desktop。
>
> 特别注意：此次安装没有full，有full的则会默认安装

（4）安装ros-kinetic-desktop

```python
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
$ sudo apt-get update
$ sudo apt-get install ros-kinetic-desktop
$ sudo rosdep init
$ rosdep update
$ echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
$ sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential
```

（5）安装一些必要ros包

```python
$ sudo apt-get install ros-kinetic-gazebo8-msgs
$ sudo apt-get install ros-kinetic-gazebo8-ros-control
$ sudo apt-get install ros-kinetic-gazebo8-plugins
$ sudo apt-get install ros-kinetic-gazebo8-ros-pkgs
$ sudo apt-get install ros-kinetic-gazebo8-ros
$ sudo apt-get install ros-kinetic-image-view
```

> 其他可能帮助信息

如果需要卸载ros的话，参考如下命令：

```python
$ sudo apt-get purge ros-*
$ sudo rm -rf /etc/ros
$ gedit ~/.bashrc
```

找到：带有kinetic的那一行删除，保存，然后：

```python
$ source ~/.bashrc
```

如果不删掉这一行或者在他之前多余的命令，那么你会在打开终端后发现第一行永远是个报错信息，虽然有时候没有什么影响。 

参考网址：[网址1](https://blog.csdn.net/qq_41058594/article/details/81079259)