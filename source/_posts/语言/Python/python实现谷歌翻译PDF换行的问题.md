---
title: python实现谷歌翻译PDF换行的问题
date: 2019-05-14 17:04:16
categories: 
- 语言
- python
tags: Python
type: "tags"
---

# 前提条件

### （1）浏览器 chrome

### （2）python编辑器：pycharm

   

# 使用说明

### 1、 pycharm上安装pyperclip、webbrowserdownloader

![logo](python实现谷歌翻译PDF换行的问题\pycharm.png)

### 2、创建google_translate.py文件，并将如下代码复制粘贴至google_translate.py文件

> 添加了百度翻译，运行这个py程序，将会在谷歌窗口弹出两个页面，分别是谷歌翻译和百度翻译

```python
#coding=utf-8
import pyperclip
import webbrowser

copyBuff = ' '
num = 1
#convinent to change num

while num == 1:
    num = num + 1
    copyedText = pyperclip.paste()
    if copyBuff != copyedText:
        copyBuff = copyedText
    normalizedText = copyBuff.replace('\n', ' ')
    url = 'https://translate.google.cn/#en/zh-CN/' + normalizedText
    webbrowser.open(url)
    url = 'https://fanyi.baidu.com/#en/zh/' + normalizedText
    webbrowser.open(url)


```
### 3、运行

#### （1）在pdf上任意选择一个段落，复制

![logo](python实现谷歌翻译PDF换行的问题\pdf.png)

#### （2）运行google_translate.py程序，chrome浏览器会自动弹出一个google翻译界面，刚才复制的内容就会直接翻译

![logo](python实现谷歌翻译PDF换行的问题\google_trans.png)

## 注：在换行出现‘-’的情况下，可能没有处理，即如下情况：

	在di-rectly这里，google翻译中没有解决，不过不影响翻译
	We present the first deep learning model to successfully learn control policies di-rectly from high-dimensional sensory input using reinforcement learning. 
