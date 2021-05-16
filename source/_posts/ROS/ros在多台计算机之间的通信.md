---
title: ros在多台计算机之间的通信
date: 2019-05-26 11:14:30
categories: ROS
tags: ROS
type: "tags"
---

# 前言

> 利用ROS的性能，能实现两台机器之间的网络通信，并且具有跨系统性（可以在不同的ros版本版本之间通信）

+ 需要两台机器在同一个局域网内
+ 这里默认已经装好ROS（可以为不同版本）

# 一、两台电脑通信前的准备工作

## 1、查看两台电脑各自的用户名和IP信息

> 在终端输入`who`能查看用户名，输入`ifconfig`能查看IP信息

如：我这里的两台服务器的用户名和IP分别是

+ cug_local，192.168.1.57
+ cug_master，192.168.1.58

![](ros在多台计算机之间的通信\user_ip.png)

## 2、修改/etc文件夹下的hosts文件

> 修改的目的是将两台电脑的ip和用户名绑定，这样在ping对方用户名时，可以解析成功

（1）修改权限

```
sudo chmod a+w /etc/hosts
```

（2）在/etc/hosts最后两行添加代码

```
sudo gedit /etc/hosts
```

添加的第一行是本机的IP和用户名

添加的第二行是另一台机器的IP和用户名

![](ros在多台计算机之间的通信\etc_host.png)

（3）重启网络

```
sudo /etc/init.d/networking restart
```

如果无法重启网络，可以参考该[网址](http://note.youdao.com/noteshare?id=fafc2918f2d674a47aeea10a8c58af88&sub=2464807983794E51824D606CCA01AEA7)

> <font color="red">两台电脑都做上面三个步骤操作</font>

# 二、两台电脑间通信测试

## 1、安装chrony

> 两台电脑上都安装chrony包，用于实现同步

```
sudo apt-get install chrony
```

## 2、安装ssh服务端

> 两台电脑上都安装ssh服务端（默认ubuntu系统自带ssh客户端）

```
sudo apt-get install openssh-server
```

> 服务端启动测试

```
ps -e |grep ssh
```

如果看到了sshd，说明ssh-server已经启动成功

## 3、ping测试

（1）cug_local机器ping 机器cug_master

```
ssh cug_local
ping cug_master
```

如果出现如下信息，则通信正常

![](ros在多台计算机之间的通信\A_ping_B.png)

（2）反向测试，cug_master机器ping机器cug_local

```
ssh cug_master
ping cug_local
```

![](ros在多台计算机之间的通信\B_ping_A.png)

# 三、~/.bashrc配置

> 说明：假设将cug_master机器当做主机master

> （1）在cug_local机器的~/.bashrc文件中添加如下两行代码

```
export ROS_HOSTNAME=cug_local
export ROS_MASTER_URI=http://cug_master:11311
```

添加完后，需要source一下

```
source ~/.bashrc
```

> （2）在cug_master机器的~/.bashrc文件中添加如下两行代码

```
export ROS_HOSTNAME=cug_master
export ROS_MASTER_URI=http://cug_master:11311
```

添加完后，需要source一下

```
source ~/.bashrc
```

# 四、ros通信

> 说明：假设将cug_master机器当做主机master

## 1、在cug_master机器上执行如下命令

（1）打开一个新终端，输入roscore

（2）打开另一个新终端，输入如下命令

```
 rosrun rospy_tutorials listener.py
```

## 2、在cug_local机器上执行如下命令

打开一个新终端，输入如下命令

```
 rosrun rospy_tutorials talker.py
```

## 3、结果显示

（1）cug_master机器上显示（rosrun的终端上）

![](ros在多台计算机之间的通信\listener.png)

（2）cug_local机器上显示（rosrun的终端上）

![](ros在多台计算机之间的通信\talker.png)

如上两图显示，则整个配置成功

# 五、架构图

![](ros在多台计算机之间的通信\communicate.png)

# 参考网址

+ [网址1](https://blog.csdn.net/heyijia0327/article/details/42065293)

+ [网址2](https://blog.csdn.net/heyijia0327/article/details/42080641)