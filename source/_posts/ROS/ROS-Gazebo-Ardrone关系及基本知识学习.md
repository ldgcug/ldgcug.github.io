---
title: ROS+Gazebo+Ardrone关系及基本知识学习
date: 2019-05-25 20:52:40
categories: ROS
tags: 
- ROS
- Gazebo
type: "tags"
---

# 前言

> 主要记录ros的一些基本指令和gazebo及ardrone的一些名词解释和相互关系

## 1、名词解释

+ ROS：ROS(Robot Operating System, 机器人操作系统)是一个适用于机器人的开源的元操作系统。它提供了操作系统应有的服务：如硬件抽象、设备驱动、函数库、可视化工具、消息传递和软件包管理等诸多功能。

+ Gazebo：可以主要用来进行机器人动力学的仿真。

+ Ardrone：四轴飞行器，支持ROS系统。

## 2、相互关系

![](ROS-Gazebo-Ardroneg关系及基本知识学习\relation.png)

## 3、Gazebo基本组成部分

![](ROS-Gazebo-Ardrone关系及基本知识学习\gazebo_zucheng.png)

![](ROS-Gazebo-Ardrone关系及基本知识学习\gazebo_zucheng2.png)

## 4、ROS基本概念

![](ROS-Gazebo-Ardrone关系及基本知识学习\ros.png)

+ 节点（node）：一个节点即为一个可执行文件，它可以通过ROS与其他节点进行通信

  例子：咱们有一个机器人，和一个遥控器，那么这个机器人和遥控器开始工作后，就是两个节点。遥控器起到了下达指 令的作用；机器人负责监听遥控器下达的指令，完成相应动作。从这里我们可以看出，<font color="red">节点是一个能执行特定工作任 务的工作单元，并且能够相互通信，从而实现一个机器人系统整体的功能。</font>在这里我们把<font color="red">遥控器和机器人简单定义为两个节点</font>，实际上在机器人中根据控制器、传感器、执行机构等不同组成模块，还可以将其进一步细分为更多的节点，这个是根据用户编写的程序来定义的。

+ 消息（message）：消息是一种ROS数据类型，用于订阅或发布到一个话题。

  消息是一种数据结构，支持多种数据类型（整形、浮点、布尔型、数组等），同时也支持消息的嵌套定义。ROS提供了大量的系统默认消息供用户使用，如geometry_msgs、sensor_msgs等，同时也支持用户定义专属数据结构的消息类型。

+ 话题（Topic）：节点可以发布消息到话题，也可以订阅话题以接收消息。

  话题是消息的载体，作用是用不同的名称区分不同消息。

  话题与消息是紧密联系在一起的。话题就像公交车，消息是公交车里装的人。公交车里可以没有人（话题上没有有效消息），但能装什么人一定会预先指定（话题一定有类型）。整个公交网络中线路名称不能重复（话题名称不能重复），要是真有两个话题名称相同类型也相同，ROS不会对其中的数据做区分，这种冲突是没有提示的。

  <font color="red">订阅/发布话题是不同步的，发布的人只管说话，订阅的人只管偷听，发布的人连续说了100句话，这100句话会排成一个队列，偷听的人要一句一句听，哦，对了，偷听的人可能不止一个</font>

+ 服务（service）：服务是应答响应模式下的信息交互方式。这种方式是基于客户端/服务器模型的。

  与话题不同的是，当服务端收到服务请求后，会对请求做出响应，将数据的处理结果返回给客户端。这种模式更适用于双向同步的信息传输，在同一个ROS网络中节点指定服务名称时不能重名。当节点A找节点B借钱时，整个网络里只有一个B，谁要是冒充B借了钱，那他就是2B。

+ master：节点管理器，ROS名称服务 (比如帮助节点找到彼此)。

  master是整个ROS运行的核心，它主要的功能就是登记注册节点、服务和话题的名称，并维护一个参数服务器。没有它你就甭想启动任何一个节点，roscore就是用来启动master的。

参考网址：[ROS官网](http://wiki.ros.org/ROS/Tutorials)、[转载](https://blog.csdn.net/lingchen2348/article/details/86134572)

## 5、ROS Ardrone常用命令

+ 起飞：

  ```
  rostopic pub -1 /ardrone/takeoff std_msgs/Empty
  ```

+ 降落

  ```
  rostopic pub -1 /ardrone/land std_msgs/Empty
  ```

+ 切换相机

  ```
  rosservice call /ardrone/togglecam
  ```

+ 获取前置相机图像

  ```
  rosrun image_view image_view image:=/ardrone/front/image_raw
  ```

+ 获取下置相机图像

  ```
  rosrun image_view image_view image:=/ardrone/bottom/image_raw
  ```

## 6、ROS常用信息显示

（1）rostopic list ：显示所有的话题信息

（2）rostopic echo [topic] ：显示发布的话题的数据信息

  如：rostopic echo /ardrone/navdata

（3）rostopic type [topic]：返回发布的话题的消息类型

![](ROS-Gazebo-Ardrone关系及基本知识学习\ros_message.png)

（4）rostopic show ardrone_autonomy/Navdata：显示ardrone_autonomy/Navdata

参考网址：[ros官网](http://wiki.ros.org/ROS/Tutorials/UnderstandingTopics)