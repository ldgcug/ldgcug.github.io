---
title: erle_copter仿真安装笔记
date: 2019-05-28 23:08:42
categories: ROS
tags: 
- ROS
- Gazebo
type: "tags"
---

## 前言

> Erle_copter仿真无人机飞行器相比于ardone仿真无人机飞行器的优势在于，Erle_copter更偏底层控制一点，使用Erle_copter仿真，可以和自己组装的真机无人机相匹配，而ardrone仿真无人机飞行器更接近于真机的ardrone或bebop，如果想要对真机ardrone或真机bebop进行修改，会有一定的难度，且购买成品无人机比自己组装无人机价格更贵。

## 安装前提

+ 操作系统：ubuntu16
+ ROS：kinetic
+ Gazebo：8.6

## 一、安装gazebo8

```python
$ sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
$ wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install gazebo8
$ sudo apt-get install libgazebo8-dev
```

## 二、安装ros kinetci（不安装desktop-full版本）

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

## 三、安装必要ROS程序包

```python
$ sudo apt-get install ros-kinetic-gazebo8-msgs
$ sudo apt-get install ros-kinetic-gazebo8-ros-control
$ sudo apt-get install ros-kinetic-gazebo8-plugins
$ sudo apt-get install ros-kinetic-gazebo8-ros-pkgs
$ sudo apt-get install ros-kinetic-gazebo8-ros
$ sudo apt-get install ros-kinetic-image-view

$ sudo apt-get install ros-kinetic-mavlink
$ sudo apt-get install ros-kinetic-octomap-msgs
$ sudo apt-get install libgoogle-glog-dev protobuf-compiler ros-$ROS_DISTRO-octomap-msgs ros-$ROS_DISTRO-octomap-ros ros-$ROS_DISTRO-joy
$ sudo apt-get install libtool automake autoconf libexpat1-dev
$ sudo apt-get install ros-kinetic-mavros-msgs
$ sudo apt-get install ros-kinetic-gazebo-msgs
```

## 四、安装erle_copter仿真环境

> 安装基础包

```python
$ sudo apt-get update
$ sudo apt-get install gawk make git curl cmake -y
```

> 安装MAVProxy依赖

```python
$ sudo apt-get install g++ python-pip python-matplotlib python-serial python-wxgtk2.8 python-scipy python-opencv python-numpy python-pyparsing ccache realpath libopencv-dev -y
```

如果安装python-wxgtk2.8报该错误：**E: Package 'python-wxgtk2.8' has no installation candidate**

则按下面方法即可解决

```python
$ sudo add-apt-repository ppa:nilarimogard/webupd8
$ sudo apt-get update
$ sudo apt-get install python-wxgtk2.8
```

> 安装MAVProxy

```python
$ sudo pip install future
$ sudo apt-get install libxml2-dev libxslt1-dev -y
$ sudo pip2 install pymavlink catkin_pkg --upgrade
$ sudo pip install MAVProxy==1.5.2
```

> 下载相关程序包

```
$ git clone https://github.com/ldgcug/erlecopter_gazebo8.git
```

> 安装ArUco

```python
$ cp -r ~/erlecopter_gazebo8/aruco-1.3.0/ ~/Downloads/
$ cd ~/Downloads/aruco-1.3.0/build
$ cmake ..
$ make
$ sudo make install

说明：如果 cmake ..  或 make 等报错，则删除build文件，重新创建build文件并编译，具体操作如下：
$ cd ~/Downloads/aruco-1.3.0/
$ rm -rf build/
$ mkdir build && cd build
$ cmake ..
$ make
$ sudo make install
```

> 下载ardupilot到特定文件夹

```python
$ mkdir -p ~/simulation; cd ~/simulation
$ git clone https://github.com/erlerobot/ardupilot -b gazebo
```

> 创建ros工作空间及初始化工作空间

```python
$ mkdir -p ~/simulation/ros_catkin_ws/src
$ cd ~/simulation/ros_catkin_ws/src
$ catkin_init_workspace
$ cd ~/simulation/ros_catkin_ws
$ catkin_make
$ source devel/setup.bash
```

> 拷贝相关源码到工作空间内

```python
$ cp -r ~/erlecopter_gazebo8/ardupilot_sitl_gazebo_plugin/  ~/simulation/ros_catkin_ws/src/ 
$ cp -r ~/erlecopter_gazebo8/hector_gazebo/  ~/simulation/ros_catkin_ws/src/ 
$ cp -r ~/erlecopter_gazebo8/rotors_simulator/ ~/simulation/ros_catkin_ws/src/ 
$ cp -r ~/erlecopter_gazebo8/mav_comm/ ~/simulation/ros_catkin_ws/src/ 
$ cp -r ~/erlecopter_gazebo8/glog_catkin/ ~/simulation/ros_catkin_ws/src/ 
$ cp -r ~/erlecopter_gazebo8/catkin_simple/ ~/simulation/ros_catkin_ws/src/ 
$ cp -r ~/erlecopter_gazebo8/mavros/ ~/simulation/ros_catkin_ws/src/
$ cp -r ~/erlecopter_gazebo8/gazebo_ros_pkgs/ ~/simulation/ros_catkin_ws/src/

添加Python和C++样例
$ cp -r ~/erlecopter_gazebo8/gazebo_cpp_examples/ ~/simulation/ros_catkin_ws/src/
$ cp -r ~/erlecopter_gazebo8/gazebo_python_examples/ ~/simulation/ros_catkin_ws/src/
```

> 拷贝fix-unused-typedef-warning.patch文件到工作空间内

```python
$ cp -r ~/erlecopter_gazebo8/fix-unused-typedef-warning.patch ~/simulation/ros_catkin_ws/src/
```

> 安装drcsim7（ubuntu16不支持apt-get，需使用源码下载安装）

```python
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
$ wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
$ sudo apt-get update

# Install osrf-common's dependencies
$ sudo apt-get install -y cmake               \
                        debhelper           \
                        ros-kinetic-ros      \
                        ros-kinetic-ros-comm
                        
# Install sandia-hand's dependencies
$ sudo apt-get install -y ros-kinetic-xacro        \
                        ros-kinetic-ros          \
                        ros-kinetic-image-common \
                        ros-kinetic-ros-comm     \
                        ros-kinetic-common-msgs  \
                        libboost-dev            \
                        avr-libc                \
                        gcc-avr                 \
                        libqt4-dev
                        
 # Install gazebo-ros-pkgs
 $ sudo apt-get install -y libtinyxml-dev                 \
                        ros-kinetic-opencv3             \
                        ros-kinetic-angles              \
                        ros-kinetic-cv-bridge           \
                        ros-kinetic-driver-base         \
                        ros-kinetic-dynamic-reconfigure \
                        ros-kinetic-geometry-msgs       \
                        ros-kinetic-image-transport     \
                        ros-kinetic-message-generation  \
                        ros-kinetic-nav-msgs            \
                        ros-kinetic-nodelet             \
                        ros-kinetic-pcl-conversions     \
                        ros-kinetic-pcl-ros             \
                        ros-kinetic-polled-camera       \
                        ros-kinetic-rosconsole          \
                        ros-kinetic-rosgraph-msgs       \
                        ros-kinetic-sensor-msgs         \
                        ros-kinetic-trajectory-msgs     \
                        ros-kinetic-urdf                \
                        ros-kinetic-dynamic-reconfigure \
                        ros-kinetic-rosgraph-msgs       \
                        ros-kinetic-tf                  \
                        ros-kinetic-cmake-modules  
                        
# Install drcsim's dependencies   
$ sudo apt-get install -y cmake debhelper                         \
                     ros-kinetic-std-msgs ros-kinetic-common-msgs   \
                     ros-kinetic-image-common ros-kinetic-geometry  \
                     ros-kinetic-robot-state-publisher            \
                     ros-kinetic-image-pipeline                   \
                     ros-kinetic-image-transport-plugins          \
                     ros-kinetic-compressed-depth-image-transport \
                     ros-kinetic-compressed-image-transport       \
                     ros-kinetic-theora-image-transport           \
                     ros-kinetic-ros-controllers                  \
                     ros-kinetic-moveit-msgs                      \
                     ros-kinetic-joint-limits-interface           \
                     ros-kinetic-transmission-interface           \
                     ros-kinetic-laser-assembler        
                     
$ sudo apt-get install ros-kinetic-pr2-controllers
```

> 拷贝drcsim相关包到工作空间

```python
$ cp -r ~/erlecopter_gazebo8/osrf-common/  ~/simulation/ros_catkin_ws/src/
$ cp -r ~/erlecopter_gazebo8/sandia-hand/ ~/simulation/ros_catkin_ws/src/
$ cp -r ~/erlecopter_gazebo8/drcsim/ ~/simulation/ros_catkin_ws/src/
```

> source一下

```python
$ source /opt/ros/kinetic/setup.bash
```

> 修改 has_binary_operator.hpp文件（为避免包BOOST_JOIN错误）

```python
$ sudo gedit /usr/include/boost/type_traits/detail/has_binary_operator.hpp
```

点击[此处](https://ldgyyf.cn/2019/05/28/ROS/erle-copter仿真安装笔记之has-binary-operator-hpp/#more)，拷贝其中的has_binary_operator.hpp代码，并粘贴至当前的has_binary_operator.hpp 文件中

主要的修改是在源文件中的两处位置添加了 `#ifndef Q_MOC_RUN` 和`#endif`

> 下载相应包进行替换（替换掉原工作空间的drcsim、hector_gazebo、gazebo_ros_pkgs） 链接: <https://pan.baidu.com/s/1TufCNJ8z5TxyC5rnZhi56A> 提取码: usjz 下载文件主要包含三个文件，分别是drcsim、hector_gazebo、gazebo_ros_pkgs，将它们解压，并将（drcsim、hector_gazebo、gazebo_ros_pkgs）复制到~/simulation/ros_catkin_ws/src目录下，将前面的三个文件进行替换

> 编译工作空间

```python
$ cd ~/simulation/ros_catkin_ws
$ catkin_make --pkg mav_msgs mavros_msgs gazebo_msgs
$ source devel/setup.bash
$ catkin_make -j 4
```

>  下载gazebo模型

```python
$ mkdir -p ~/.gazebo/models
$ git clone https://github.com/erlerobot/erle_gazebo_models
$ mv erle_gazebo_models/* ~/.gazebo/models
```

## 五、启动erle_copter

> 启动ArduCopter（一个终端）

```python
$ source ~/simulation/ros_catkin_ws/devel/setup.bash
$ cd ~/simulation/ardupilot/ArduCopter
$ ../Tools/autotest/sim_vehicle.sh -j 4 -f Gazebo
```

> 在另一个终端启动launch

```python
$ cd ~/simulation/ros_catkin_ws/
$ source ~/simulation/ros_catkin_ws/devel/setup.bash
$ roslaunch ardupilot_sitl_gazebo_plugin erlecopter_spawn.launch
```

> 在第一个终端上输入如下命令：

```python
$ param load /[path_to_your_home_directory]/simulation/ardupilot/Tools/Frame_params/Erle-Copter.param
```

用你的实际目录替换掉上面的path_to_your_home_directory，如我的是：param load /home/cug/simulation/ardupilot/Tools/Frame_params/Erle-Copter.param

> 起飞测试，仍然在第一个终端执行

```python
$ mode GUIDED
$ arm throttle
$ takeoff 2
```

说明：在执行了arm throttle后，尽快输入takeoff 2 已完成起飞

## 官网安装网址：

> <http://docs.erlerobotics.com/simulation/configuring_your_environment>

## Github安装网址：

> <https://github.com/ldgcug/erlecopter_gazebo8/blob/master/README.md>

## 总结

> 在前面的安装过程中，中间有一个步骤是从百度云盘上下载了三个文件夹，并将之前的工作空间内的相关文件夹进行替换，之所以这样做，是因为在将文件上传到github的过程中，好像有部分文件丢失，因此下载了这缺失的部分文件，在编译过程中会报错，因此将文件解压并上传到百度云盘上以解决该问题
>
> 第二个是还是从别人的github上git程序包，原因如上，但是也压缩上传到百度云，下载后替换掉没有换，由于这个不在工作空间内，因此不影响编译，但是在运行时，会报权限的相关错误。因此还是从别人的github上下载较好。