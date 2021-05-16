---
title: gazebo配置
date: 2019-05-24 23:16:55
categories: ROS
tags: 
- ROS
- Gazebo
type: "tags"
---

# 前言

> 记录的是研究生阶段以来使用gazebo过程中遇到过的一些坑，以及一些相关的gazebo配置

## 1、gazebo模型文件说明

> 如果不是安装的gazebo8，如默认安装的ros kinetic版本对应的gazebo7和ros indigo版本对应的gazebo2的话，在终端输入gazebo的时候，会卡着，gazebo界面出不来。其原因是因为没有下载gazebo的模型，下载完后将模型拷贝到~/.gazebo/models/文件夹即可。
>
> 值得注意的是：~/.gazebo/models/文件夹下只需要有sun模型文件和ground_plane模型文件就能正常打开gazebo界面

## 2、gazebo_model_path配置

> 针对以往的总是要将gazebo model文件放在~/.gazebo/models文件，而不能存放在自己下载的ros程序包下的model文件夹下，主要是因为环境变量没有配置，配置好后就可以将所有的模型文件放在自己想放的位置。

（1）首先在~/.bashrc最后一行添加gazebo_model_path路径

```python
export GAZEBO_MODEL_PATH="/home/cugrobot/catkin_ws/src/ardrone_simulator_gazebo7/cvg_sim_gazebo/models"
```

![](gazebo配置\gazebo_model_path.png)

其对应的模型文件夹如下：

![](gazebo配置\gazebo_models.png)

（2）然后，在launch启动文件夹下，添加env代码，如下

![](gazebo配置\launch.png)

通过`env | grep GAZEBO_MODEL_PATH`命令可以查看其配置路径，同理可以应用于其他路径查看

## 3、gazebo关闭client界面

> 在做强化学习训练时，打开gazebo界面可能会使训练比较耗时，因此关闭client界面也许是一种比较好的方法。

gazebo平台第三视角关闭方法如下：

```python
$ roscd gazebo_ros
$ cd launch
$ sudo gedit empty_world.launch
```

在打开的界面中修改两处地方，第一处在第7行，将<arg name="gui" default="true"/>其中的true改为false；第二处在第41行，有一个  <!-- start gazebo client   -->的注释，将注释范围扩大，将42-44行全部注释掉，即该后面的整个group注释。

> 若上面方法还不能关闭界面，在launch启动文件里面，找到所有相关联的启动文件，将上面的修改方法在launch里面也执行一遍。

## 4、gazebo仿真世界中模型位置修改

### 4.1 通过os.system()函数实现rosservice服务

> 我们能在终端通过调用 /gazebo/set_model_state服务来重置pose和twist，这是最初始的时候的方法，但在测试过程中，重置位置容易出现在限定范围外，因此不好，关于修改模型位置方法可以参考`4.2节`

```
rosservice call /gazebo/set_model_state '{model_state: { model_name: quadrotor, pose: { position: { x: 5, y: 0 ,z: 1 }, orientation: {x: 0, y: 0.491983115673, z: 0, w: 0.870604813099 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }'
```

其中，需要说明的是quadrotor是你要修改的模型名称

参考网址：<http://wiki.ros.org/simulator_gazebo/Tutorials/Gazebo_ROS_API>

> 如果要在python程序中调用上面的重置命令，则需要使用os.system（）函数来实现

```
os.system('''rosservice call /gazebo/set_model_state "[quadrotor, [[5, 0, 0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], '']"'''
```

如果写成动态加载代码的话，可以如下图所示：

![](gazebo配置\reset_pose.png)

参考网址：<https://blog.csdn.net/lordofrobots/article/details/78088517?utm_source=debugrun&utm_medium=referral>

### 4.2 rospy.ServiceProxy()函数实现位置修改和获取模型位置

>  重置模型坐标位置主要是调用`/gazebo/set_model_state`服务，而获取模型坐标位置是调用`/gazebo/get_model_state`服务

其中，如`/gazebo/set_model_state`的`type`类型可以通过`rosservice info /gazebo/set_model_state`命令来确定

具体代码如下：

```python
#!/usr/bin/env python
#coding=utf8
import rospy
from gazebo_msgs.srv import *

# 重置无人机坐标位置
def set_model_pos():
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state_service = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
    objstate = SetModelStateRequest()

    #set quadrotor pose
    objstate.model_state.model_name = 'quadrotor'
    objstate.model_state.pose.position.x = 5
    objstate.model_state.pose.position.y = 0
    objstate.model_state.pose.position.z = 0
    objstate.model_state.pose.orientation.w = 1
    objstate.model_state.pose.orientation.x = 0
    objstate.model_state.pose.orientation.y = 0
    objstate.model_state.pose.orientation.z = 0
    objstate.model_state.twist.linear.x = 0.0
    objstate.model_state.twist.linear.y = 0.0
    objstate.model_state.twist.linear.z = 0.0
    objstate.model_state.twist.angular.x = 0.0
    objstate.model_state.twist.angular.y = 0.0
    objstate.model_state.twist.angular.z = 0.0
    objstate.model_state.reference_frame = "world"

    result = set_state_service(objstate)
    
# 获取无人机坐标位置
def get_model_pos():
    get_state_service = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
    model = GetModelStateRequest()
    model.model_name = 'quadrotor'
    objstate = get_state_service(model)
    state = (objstate.pose.position.x,objstate.pose.position.y,objstate.pose.position.z)
    print('pos',state) 
```

该代码转载于[此处](https://blog.csdn.net/penge666/article/details/87900911)

## 5、gazebo仿真时间加速

> 主要遇到的问题是在进行DQN训练时，由于机器性能及每步飞行时间等原因，训练时长较久，因此能够在仿真中修改一些配置，使得仿真的时间比现实时间更快。

主要是通过修改world文件里面的一些物理属性，来实现仿真时间加速的效果，若需要加速时，还是在重新测试比较好，我修改为如下代码后，较之前能有3倍左右的提升速度，并且没有使用rospy.sleep(2)函数，而是使用的rospy.Rate(0.46).sleep()来代替。

```
    <physics name='default_physics' default='0' type='ode'>
      <real_time_update_rate>0</real_time_update_rate>
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
```

通过修改max_step_size的值能对gazebo进行加速，一般修改的时候real_time_update_rate设置为0。

Max_step_size:0.001（默认值）

Real_time_update_rate:1（默认值）

Real_time_update_rate:1000（默认值）

![](gazebo配置\gazebo_properties.png)

参考网址：<http://gazebosim.org/tutorials?tut=modifying_world&cat=build_world>

## 6、gazebo添加定制模型

> 以前都是自己制作的一些简单模型导入到gazebo仿真世界中，但想加载一些特定的模型时，不一定能自己制作出来，这时可以下载[3D Warehouse](https://3dwarehouse.sketchup.com/)网站上做好的模型，导入到gazebo仿真直接中即可

例如，这样的模型则不一定能自己制作出来

![](gazebo配置\gate.png)

因此，我们可以在[3D Warehouse](https://3dwarehouse.sketchup.com/)上搜索关键词，然后找到想要的模型，点击进去后，以Collada File文件形式下载即可。

![](gazebo配置\collada.png)

（1）下载完后，解压，解压后更改dae的名字。然后在gazebo_model_path的文件夹目录下创建相对应的模型文件，主要包括model.config、model.sdf文件和mesh文件夹（文件夹下只有dae文件）

![](gazebo配置\model.png)

其中，model.sdf文件内容如下：

```
<?xml version="1.0" ?>
<sdf version="1.4">
  <model name="Arc">
    <pose>0 5 0 0 0 0</pose>
    <static>true</static>
    <link name="link">
      <inertial>
        <mass>0.1</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>10 10 0.2</size>
          </box>
        </geometry>
      </collision>
   
      <visual name="visual">
        <geometry>
          <mesh>
            <!--uri>model://marker/meshes/artag_01.dae</uri-->
            <uri>model://arc/meshes/arc.dae</uri>
            <!-- <scale> 0.01 0.01 0.01 </scale>-->
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
```

（2）测试

创建arc.world文件，并将如下内容拷贝进去

```
<?xml version="1.0"?>
<sdf version="1.4">
  <world name="default">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
<!--
    <model name="arc">
      <pose>0 0 0  0 0 0</pose>
      <static>true</static>
      <link name="body">
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>file://arc.dae</uri>
              <scale> 0.03 0.03 0.03</scale>
            </mesh>

          </geometry>
        </visual>
      </link>
    </model>
-->
    <model name='arc'>
      <static>1</static>
      <link name='arc_link'>
        <pose frame=''>0 0 0 0 -0 0</pose>
        <collision name='collision'>
          <geometry>
            <mesh>
              <uri>model://arc/meshes/arc.dae</uri>
              <scale>0.03 0.03 0.03</scale>
            </mesh>
          </geometry>
          
        </collision>
        <visual name='visual'>
          <geometry>
            <mesh>
              <uri>model://arc/meshes/arc.dae</uri>
              <scale>0.03 0.03 0.03</scale>
            </mesh>
          </geometry>
        </visual>
        <gravity>1</gravity>
      </link>
      <pose frame=''>0 0 0.05 0 0 0</pose>
    </model>
  </world>
</sdf>
```

> 说明：上面代码中注销掉的部分如 <uri>file://arc.dae</uri>是以文件形式导入，而后面<uri>model://arc/meshes/arc.dae</uri>是以sdf形式导入。
>
> 需要注意的是：需要使用<scale>0.03 0.03 0.03</scale>来对模型进行调节大小

（3）显示

运行gazebo arc.world

![](gazebo配置\gazebo_show.png)

参考网址：<http://gazebosim.org/tutorials?tut=import_mesh#PreparetheMesh>

<https://answers.ros.org/question/42529/how-to-import-collada-dae-files-into-gazebo-rosfuerte/>

## 7、Gazebo场景纹理图重置

> 主要是我在做强化学习（DQN）训练的过程中，需要将纹理时常更换，因此在网上查找相关教程，最终实现了gazebo的场景重置

先看代码，后面在进行解释，该代码是从训练的代码中截取的部分，需要的内容全部都在了

```python
#!/usr/bin/env python
#coding=utf8

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import *
import rospy

import roslib;roslib.load_manifest('test')

class DQN():
    def __init__(self):
        self.pubModelStates = rospy.Subscriber('gazebo/model_states',ModelStates,self.get_model_pos)
        self.database_model_name = \
        ["asphalt1","asphalt2","asphalt3","asphalt4",
        "brick1","brick2","brick3","brick4",
        ]

    #重置特定模型位置
    def set_model_pos(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state_service = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        objstate = SetModelStateRequest()

        #set quadrotor pose
        objstate.model_state.model_name = 'quadrotor'
        objstate.model_state.pose.position.x = 5
        objstate.model_state.pose.position.y = 0
        objstate.model_state.pose.position.z = 0
        objstate.model_state.pose.orientation.w = 1
        objstate.model_state.pose.orientation.x = 0
        objstate.model_state.pose.orientation.y = 0
        objstate.model_state.pose.orientation.z = 0
        objstate.model_state.twist.linear.x = 0.0
        objstate.model_state.twist.linear.y = 0.0
        objstate.model_state.twist.linear.z = 0.0
        objstate.model_state.twist.angular.x = 0.0
        objstate.model_state.twist.angular.y = 0.0
        objstate.model_state.twist.angular.z = 0.0
        objstate.model_state.reference_frame = "world"
        
        result = set_state_service(objstate)

    #获取特定模型位置
    def get_model_pos(self):
        get_state_service = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        model = GetModelStateRequest()
        model.model_name = 'quadrotor'
        objstate = get_state_service(model)
        state = (objstate.pose.position.x,objstate.pose.position.y,objstate.pose.position.z)
        print('pos',state)
    
    #删除模型文件
    def delete_sdf_model(self):
        rospy.wait_for_service('gazebo/delete_model')
        delete_model_service = rospy.ServiceProxy('gazebo/delete_model',DeleteModel)
        objstate = DeleteModelRequest()
        objstate.model_name = "grass7_plane"
        if objstate.model_name in self.database_model_name:
            try:
                delete_model_service(objstate)
                print('delete model success')
            except Exception as e:
                print("delete model failed")

            self.spawn_sdf_model()

    #重置模型文件
    def spawn_sdf_model(self):
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_service = rospy.ServiceProxy('gazebo/spawn_sdf_model',SpawnModel)
        with open("/home/cug/qlab_ws/src/qlab/qlab/qlab_gazebo/models/asphalt1/model.sdf","r") as f:
            model_xml = f.read()
        #model_name model_xml robot_namespace initial_pose reference_frame
        objstate = SpawnModelRequest()
        objstate.model_name = "asphalt1"
        objstate.model_xml = model_xml
        objstate.robot_namespace = ""
        
        pose = Pose()
        pose.position.x = 0.222657 
        pose.position.y = -0.204052
        pose.position.z =  0
        objstate.initial_pose = pose
        objstate.reference_frame = "world"
        try:
            #spawn_model_service("asphalt1_plane",model_xml,"",pose,"world")
            spawn_model_service(objstate)
            print('spawn model success')
        except Exception as e:
            print('spawn model failed') 

if __name__ == "__main__":
    rospy.init_node('test')
    dqn = DQN()
    dqn.delete_sdf_model()
    dqn.set_model_pos()
    dqn.get_model_pos(

```

> 说明：gazebo场景重置，主要用到两个rosservice，分别是`gazebo/delete_model`和`gazebo/spawn_sdf_model`

在重置的过程中，首先需要先删除模型，即调用`gazebo/delete_model`服务，然后在重新生成，这时需要找到想要生成的模型的sdf文件所在位置，然后读取并调用`gazebo/spawn_sdf_model`服务，即可实现gazebo场景纹理图的重置

主要需要查看的帮助信息是

（1）roservice list：查找相关service服务

（2）rosservice info [service_name]：查找对应服务的数据类型

![](gazebo配置\service_type.png)

对其中一个进行解释说明：如

```python
        objstate = SpawnModelRequest()
        objstate.model_name = "asphalt1"
        objstate.model_xml = model_xml
        objstate.robot_namespace = ""
        
        pose = Pose()
        pose.position.x = 0.222657 
        pose.position.y = -0.204052
        pose.position.z =  0
        objstate.initial_pose = pose
        objstate.reference_frame = "world"
```

这里就和/gazebo/spawn_sdf_model里面的Args对应，分别是model_name、model_xml、robot_namespace、initial_pose、reference_frame

又如：

```python
#重置特定模型位置
    def set_model_pos(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state_service = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        objstate = SetModelStateRequest()

        #set quadrotor pose
        objstate.model_state.model_name = 'quadrotor'
        objstate.model_state.pose.position.x = 5
        objstate.model_state.pose.position.y = 0
        objstate.model_state.pose.position.z = 0
        objstate.model_state.pose.orientation.w = 1
        objstate.model_state.pose.orientation.x = 0
        objstate.model_state.pose.orientation.y = 0
        objstate.model_state.pose.orientation.z = 0
        objstate.model_state.twist.linear.x = 0.0
        objstate.model_state.twist.linear.y = 0.0
        objstate.model_state.twist.linear.z = 0.0
        objstate.model_state.twist.angular.x = 0.0
        objstate.model_state.twist.angular.y = 0.0
        objstate.model_state.twist.angular.z = 0.0
        objstate.model_state.reference_frame = "world"
```

这里需要设置很多position和orientation和twist是因为如下图所示包含的信息

![](gazebo配置\rosmsg_info.png)

参考网址：[网址1](https://answers.ros.org/question/246419/gazebo-spawn_model-from-py-source-code/)、[网址2](https://answers.ros.org/question/248630/calling-gazebospawn_sdf_model-service-from-rosjava/)、[网址3](http://gazebosim.org/tutorials/?tut=ros_comm)、[网址4](http://wiki.ros.org/cn/ROS/Tutorials/WritingServiceClient%28python%29)

