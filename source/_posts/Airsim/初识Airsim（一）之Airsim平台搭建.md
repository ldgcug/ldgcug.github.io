---
title: 初识Airsim（一）之Airsim平台搭建
date: 2019-06-16 21:54:08
categories: Airsim
tags: 
- Airsim
- Unreal Engine
type: "tags"
---

## 前言

> 基于Airsim，搭建一个更逼真的仿真环境，比原有的Gazebo效果更好

## 一、相关概念介绍

- Airsim：AirSim 是微软开源的一个跨平台的建立在虚幻引擎（ Unreal Engine）上的无人机以及其它自主移动设备的模拟器。 它支持硬件在循环与流行的飞行控制器的物理和视觉逼真模拟。它被开发为一个虚幻的插件，可以简单地放到任何你想要的虚幻环境中。

  该模拟器创造了一个高还原的逼真虚拟环境，模拟了阴影、反射等其它现实世界中容易干扰的环境，让无人机不用经历真实世界的风险就能进行训练。

  AirSim 的目标是作为AI研究的平台，以测试深度学习、计算机视觉和自主车辆的增强学习算法。为此， AirSim 还公开了 API，以平台独立的方式检索数据和控制车辆。

  Airsim官方Github：<https://github.com/Microsoft/AirSim> 

- Unreal Engine：Unreal是UNREAL ENGINE（虚幻引擎）的简写，由Epic开发，是目前世界知名授权最广的游戏引擎之一，占有全球商用游戏引擎80%的市场份额。

  “Unreal Engine 3”3D引擎采用了目前最新的即时光迹追踪、HDR光照技术、虚拟位移…等新技术，而且能够每秒钟实时运算两亿个多边形运算，效能是目前“Unreal Engine”的100倍，而通过nVIDIA的GeForce 6800显示卡与“Unreal Engine 3”3D引擎的搭配，可以实时运算出电影CG等级的画面，效能非常非常恐怖。

  基于它开发的大作无数，除《虚幻竞技场3》外，还包括《战争机器》、《质量效应》、《生化奇兵》等等。在美国和欧洲，虚幻引擎主要用于主机游戏的开发，在亚洲，中韩众多知名游戏开发商购买该引擎主要用于次世代网游的开发，如《剑灵》、《TERA》、《战地之王》、《一舞成名》等。 iPhone上的游戏有《无尽之剑》（1、2、3）、《蝙蝠侠》等

## 二、版本说明

> **AirSim最新版本已支持Visual Studio 2017与Unreal Engine 4.18**

- Windows10

- Visual Studio 2017（需要安装VC++ 和Windows SDK8.1）

- Unreal Engine 4.18（通过Epic Games Launcher安装）

- Git（下载Airsim1.2源码，不要用VS2017 URL链接下载）

- Airsim1.2

    

## 三、软件安装

### 3.1 **Visual Studio2017安装**

（1）点击[下载](https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads)，下载VS2017社区版

![](初识Airsim（一）之Airsim平台搭建\VS2017.png)

（2）安装VS2017

安装过程中请确保安装VC++ 和Windows SDK8.1（或Windows SDK10）的安装

![](初识Airsim（一）之Airsim平台搭建\VS and Windows SDK.png)

此外，我还安装了单个组件中的游戏和图形的部分组件

![](初识Airsim（一）之Airsim平台搭建\single_zujian.png)

> 总安装大小6个多G，耗时有点长

### 3.2 **虚幻引擎（Unreal Engine）的安装**

（1）点击[下载](https://www.unrealengine.com/zh-CN/download)，下载`Epic Games Launcher`

下载过程中，若没有注册过Epic账号的，需要先注册，然后在登录下载

（2）运行`Epic Games Launcher`，在弹出的界面中，选择`Library`，点击`+`号，选择`4.18`版本进行安装

![](初识Airsim（一）之Airsim平台搭建\Unreal Engine.png)

> 注意一定要下载4.18版本，下载了多个版本也没有关系，最后需要启动4.18版本

### 3.3 Git安装

（1）点击[下载](https://git-scm.com/downloads)，下载Git

（2）默认安装即可

## 四、**搭建Airsim环境并配置**

### 4.1 下载Airsim源码

（1）在git bash窗口输入如下命令，下载airsim源码

```
git clone https://github.com/microsoft/AirSim.git
```

下载完后，将Airsim文件存放到其他位置，如我的存放在D盘目录下的自定义新文件夹下

### 4.2 编译

（1）打开window菜单，找到Visual Studio 2017，并双击打开VS2017的x64本机命令提示

![](初识Airsim（一）之Airsim平台搭建\VS2017-x64png.png)

在打开的窗口中，进入到Airsim所在目录

（2）编译

> 首先，找到**\AirSim\AirLib\deps\eigen3\Eigen\src\Core\arch\CUDA\Half.h**文件，修改Half.h文件的“AS ls”的引号即可。如图所示

![](初识Airsim（一）之Airsim平台搭建\half-h.png)

> 之所以修改该 `“”`，是因为如果不修改，会在后面的编译过程中，碰到如下错误

![](初识Airsim（一）之Airsim平台搭建\hald-h-error.png)

>  然后，进入Airsim目录后，执行如下命令：

```
build.cmd --no-full-poly-car
```

其实也可以只执行build.cmd ，也不用添加后面的 --no-full-poly-car，之所以这样做是为了在编译过程中能节省很大的时间，在下面这一步，耗时较长，我这里大概要花20分钟

![](初识Airsim（一）之Airsim平台搭建\high-poly-car.png)

> 最后，编译成功后的界面显示如下图

![](初识Airsim（一）之Airsim平台搭建\build_success.png)

## 五、**UE4与Airsim联系起来**

### 5.1 启动Unreal Enigen 4.18

![](初识Airsim（一）之Airsim平台搭建\Unreal Engine4.18.png)

### 5.2 新建工程（c++项目），如Rolling

![](初识Airsim（一）之Airsim平台搭建\c++ project.png)

注意：名称一定要写成英文，不能用中文

### 5.3 复制文件

复制AirSim\Unreal\Plugins文件夹 到 Rolling目录下；

复制AirSim\Unreal\Environments\Blocks文件夹下的clean.bat和GenerateProjectFiles.bat 文件到 Rolling目录下；

![](初识Airsim（一）之Airsim平台搭建\copy_file.png)

> 在这个过程中，可能会重新编译C++类，会在UE工程下生成Rollings.sln

### 5.4 运行Rolling工程

在Rolling目录下，双击Rolling.sln在VS2017中打开该工程

首先修改配置：DebugGame Editor + win64

![](初识Airsim（一）之Airsim平台搭建\Debug.png)

然后点击【生成】-【重新生成解决方案】

最后，按F5键，运行工程项目

![](初识Airsim（一）之Airsim平台搭建\Rolling.png)

### 5.5 添加Quadrotor

进入  `【设置】`-`【世界设置】`，修改其中的Game mode，修改为AirSimGameMode

![](初识Airsim（一）之Airsim平台搭建\gamemode.png)

点击`【播放】`按钮，在弹出的框中选择`【不】`，即可

第一次加载界面如下图，右下角还在编译着色器，稍微等一会就好

![](初识Airsim（一）之Airsim平台搭建\quadrotor_show1.png)

等一会后，无人机就加载出来了

![](初识Airsim（一）之Airsim平台搭建\quadrotor_show2.png)

## 六、参考链接

### 6.1 安装文档参考

- [Airsim官网](https://microsoft.github.io/AirSim/docs/build_windows/)

- [知乎文档](https://zhuanlan.zhihu.com/p/52665325?utm_source=qq&utm_medium=social&utm_oi=60530387582976)

### 6.2 相关概念参考

- [Unreal Engine](https://blog.csdn.net/hhlenergystory/article/details/80275617)

- [Airsim](https://www.oschina.net/news/89300/airsim-1-1-1)

## 总结

> 至此，Airsim在Windows平台下的搭建已经基本成功，后面会继续添加新的场景，如城市场景；也会自己在去编写一些程序代码进行控制

> 我搭建的电脑机器配置如下：
>
> **i7处理器，8G RAM，GT730显卡，128G SSD**

