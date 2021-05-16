---
title: ubuntu16安装ROS
date: 2019-05-22 14:16:55
categories: ROS
tags: 
- Linux
- ROS
type: "tags"
---

> ubuntu与ros对应版本关系（我目前更多的用的是ubuntu16）

![](ubuntu16安装ROS\ubuntu_ros.png)

> ubuntu16下ros安装步骤

```python
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
$ sudo apt-get update
$ sudo apt-get install ros-kinetic-desktop-full
$ sudo rosdep init
$ rosdep update
$ echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
$ sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential
```

安装完成后，在终端输入roscore，若最后出现roscore，则说明安装成功

原文：ros[官网](http://wiki.ros.org/kinetic/Installation/Ubuntu)

> 总结：上面安装步骤安装的是ros kinetic 桌面完全版，并且安装完后，默认安装了gazebo7。但在终端输入gazebo，并不会出现gazebo界面，原因是缺少gazebo的两个最基本模型文件（sun和ground_plane）。gazebo_models文件下载，点击[此处](https://bitbucket.org/osrf/gazebo_models/src/default/)。