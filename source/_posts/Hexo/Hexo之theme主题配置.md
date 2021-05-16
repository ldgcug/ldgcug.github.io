---
title: Hexo之theme主题配置
date: 2019-05-14 23:52:09
categories: Hexo
tags: NexT
type: "tags"
---

#  说明：每一个大标题结尾都有一个参考链接，出现问题时，请参考其参考链接

#  一、下载 Hexo主题

## 1、到官网选择[自己喜欢的Hexo主题](https://hexo.io/themes/)

![logo](Hexo之theme主题配置\hexo_theme.png)

## 2、下载 NextT主题

> 在git bash窗口上，输入如下命令：

```
$ cd <博客存放的目录>
$ git clone https://github.com/iissnan/hexo-theme-next themes/next
```

> 将主题克隆到theme目录下后，会在其目录下发现多出一个next文件夹
>
> 注：若需要下载其他主题，只需要在上面的代码中，将next更改为其他主题名称即可

![](Hexo之theme主题配置\theme.png)

## 参考网址：<https://www.jianshu.com/p/33bc0a0a6e90?tdsourcetag=s_pctim_aiomsg>

  

# 二、NexT主题配置

>+ 在 Hexo中有两份主要的配置文件，其名称都是 _config.yml。其中，一份位于站点根目录下，主要包含 Hexo 本身的配置；另一份位于主题目录下，这份配置由主题作者提供，主要用于配置主题相关的选项。
>+ 为了描述方便，在以下说明中，将前者称为**站点配置文件**， 后者称为**主题配置文件**。
>+ 以下所有终端执行的命令都在你的 Hexo 根目录下
>
>

## 1、基本信息配置

> 基本信息包括：博客标题、作者、描述、语言等等。

打开 **站点配置文件** ，找到Site模块

```
title: 标题
subtitle: 副标题
description: 描述
author: 作者
language: 语言（简体中文是zh-Hans）
timezone: 网站时区（Hexo 默认使用您电脑的时区，不用写）
```

我的配置如下：

![](Hexo之theme主题配置\site.png)

## 2、菜单设置

> 菜单包括：首页、归档、分类、标签、关于等等

我们刚开始默认的菜单只有首页和归档两个，不能够满足我们的要求，所以需要添加菜单，打开 **主题配置文件** 找到`Menu Settings`

```
menu:
  home: / || home                          //首页
  archives: /archives/ || archive          //归档
  categories: /categories/ || th           //分类
  tags: /tags/ || tags                     //标签
  about: /about/ || user                   //关于
  #schedule: /schedule/ || calendar        //日程表
  #sitemap: /sitemap.xml || sitemap        //站点地图
  #commonweal: /404/ || heartbeat          //公益404
```

我的配置如下：

![](Hexo之theme主题配置\menu.png)

## 3、Next主题样式设置

打开 **主题配置文件** 找到`Scheme Settings`

```
# Schemes
# scheme: Muse
# scheme: Mist
# scheme: Pisces
scheme: Gemini
```

我选择的是Gemini风格

## 4、侧栏设置

> 侧栏设置包括：侧栏位置、侧栏显示与否、文章间距、返回顶部按钮等等

打开 **主题配置文件** 找到`sidebar`字段

```
sidebar:
# Sidebar Position - 侧栏位置（只对Pisces | Gemini两种风格有效）
  position: left        //靠左放置
  #position: right      //靠右放置

# Sidebar Display - 侧栏显示时机（只对Muse | Mist两种风格有效）
  #display: post        //默认行为，在文章页面（拥有目录列表）时显示
  display: always       //在所有页面中都显示
  #display: hide        //在所有页面中都隐藏（可以手动展开）
  #display: remove      //完全移除

  offset: 12            //文章间距（只对Pisces | Gemini两种风格有效）

  b2t: false            //返回顶部按钮（只对Pisces | Gemini两种风格有效）

  scrollpercent: true   //返回顶部按钮的百分比
```

## 5、头像设置

打开 **主题配置文件** 找到`Sidebar Avatar`字段

```
# Sidebar Avatar
avatar: /images/header.jpg
```

这是头像的路径，只需把你的头像命名为`header.jpg`（随便命名）放入`themes/next/source/images`中，将`avatar`的路径名改成你的头像名就OK啦！

![](Hexo之theme主题配置\header.png)

## 6、设置RSS（后面可以在继续操作，目前还存在问题，这里不显示操作步骤，详细看后面的参考链接）

## 7、添加分类模块

1、新建一个分类页面

```
$ hexo new page categories
```

2、你会发现你的`source`文件夹下有了`categorcies/index.md`，打开`index.md`文件将title设置为`title: 分类`

目录结构如下：

![](Hexo之theme主题配置\categories.png)

3、把文章归入分类只需在文章的顶部标题下方添加`categories`字段，即可自动创建分类名并加入对应的分类中

举个栗子：

```
title: 分类测试文章标题
categories: 分类名
```

## 8、添加标签模块

1、新建一个标签页面

```
$ hexo new page tags
```

2、你会发现你的`source`文件夹下有了`tags/index.md`，打开`index.md`文件将title设置为`title: 标签`

3、把文章添加标签只需在文章的顶部标题下方添加`tags`字段，即可自动创建标签名并归入对应的标签中

举个栗子：

```
title: 标签测试文章标题
tags: 
  - 标签1
  - 标签2
  ...
```

## 9、添加关于模块

1、新建一个关于页面

```
$ hexo new page about
```

2、你会发现你的`source`文件夹下有了`about/index.md`，打开`index.md`文件即可编辑关于你的信息，可以随便编辑。

## 10、添加搜索功能

1、安装 [hexo-generator-searchdb](https://link.jianshu.com/?t=https%3A%2F%2Fgithub.com%2Fflashlab%2Fhexo-generator-search) 插件

```
$ npm install hexo-generator-searchdb --save
```

2、打开 **站点配置文件** 找到`Extensions`在下面添加

```
# 搜索
search:
  path: search.xml
  field: post
  format: html
  limit: 10000
```

我的配置如下：

![](Hexo之theme主题配置\search.png)

3、打开 **主题配置文件** 找到`Local search`，将`enable`设置为`true`

## 11、设置后博客界面

![](Hexo之theme主题配置\blog.png)

## 参考链接：<https://www.jianshu.com/p/3a05351a37dc?tdsourcetag=s_pctim_aiomsg>

  

# 三、访问统计及数字统计

## 1、数字统计

> 显示文章字数统计、阅读时长、总字数

+ 安装插件

```
$ npm i --save hexo-wordcount
```



> 在 **主题配置文件** 中，搜索关键字 `post_wordcount`

```
# Post wordcount display settings
# Dependencies: https://github.com/willin/hexo-wordcount
post_wordcount:
  item_text: true
  #字数统计
  wordcount: true
  #预览时间
  min2read: true
  #总字数,显示在页面底部
  totalcount: false
  separated_meta: true
```

## 2、访问统计

> ### LeabCloud - 文章阅读量

- 注册 [LeabCloud](https://leancloud.cn/)，`访问控制台` ，`创建应用` ，新应用名称可任意填写，选择“开发板”创建应用。创建完成之后点击新创建的应用的名字来打开应用参数配置界面，并点击点击左侧右上角的齿轮图标，新建Class，如下图所示：

![](Hexo之theme主题配置\leabclound.png)

+ 创建完成之后，左侧数据栏应该会多出一栏名为 `Counter` 的栏目，这个时候我们点击设置，切换到test应用的操作界面。在弹出的界面中，选择左侧的 `应用Key` 选项，即可发现我们创建应用的 `AppID` 以及 `AppKey` ，有了它，我们就有权限能够通过主题中配置好的Javascript代码与这个应用的Counter表进行数据存取操作了。

![](Hexo之theme主题配置\count.png)

- 在 **主题配置文件** 中，搜索关键字 `leancloud_visitors` ，将 `false` 改为 `true` ，并复制粘贴上述的 `AppID`以及 `AppKey`

![](Hexo之theme主题配置\theme_visitiors.png)

需要特别说明的是：记录文章访问量的唯一标识符是文章的发布日期以及文章的标题，因此请确保这两个数值组合的唯一性，如果你更改了这两个数值，会造成文章阅读数值的清零重计。

- Web 安全。因为 AppID 以及 AppKey 是暴露在外的，因此如果一些别有用心之人知道了之后用于其它目的是得不偿失的，为了确保只用于我们自己的博客，建议开启 Web 安全选项，这样就只能通过我们自己的域名才有权访问后台的数据了，可以进一步提升安全性。

选择应用的设置的 `安全中心` 选项卡:

在 `Web 安全域名` 中填入我们自己的博客域名，来确保数据调用的安全。

## 3、图形显示

![](Hexo之theme主题配置\count_show.png)

## 参考链接：<http://dinghongkai.com/2017/12/19/Blog-development-5-NexT-Theme-Advanced-Customization/>