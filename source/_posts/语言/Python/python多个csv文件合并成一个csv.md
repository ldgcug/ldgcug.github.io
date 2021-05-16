---
title: python多个csv文件合并成一个csv
date: 2019-05-26 23:20:52
categories: 
- 语言
- python
tags: Python
type: "tags"
---

# 前言

> 在训练过程中，会产生多个txt转的csv文件，最后需要合并成一个完整的csv。

```
# -*- coding:utf8 -*-
import glob
import time

csvx_list = glob.glob('*.csv')
print('总共发现%s个CSV文件'% len(csvx_list))
time.sleep(2)
print('正在处理............')
for i in csvx_list:
    fr = open(i,'r').read()
    with open('csv_to_csv.csv','a') as f:
        f.write(fr)
    print('写入成功！')
print('写入完毕！')
print('10秒钟自动关闭程序！')
time.sleep(10)
```

