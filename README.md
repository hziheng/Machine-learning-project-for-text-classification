# ML_code completes


- [ML\_code completes](#ml_code-completes)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Installing](#installing)
  - [Usage ](#usage-)
  - [参数介绍](#参数介绍)
  - [文件目录介绍](#文件目录介绍)
  - [开发日志](#开发日志)

## About <a name = "about"></a>

This is a machine learning code template library, which contains various examples of common machine learning and deep learning. I wish users could use the various models in it by simply changing the input path to the file (provided the file is processed in the same format as I provided), and the parameters of the network will be provided in a summary file, which is convenient for users to adjust parameters. In short, the idea behind this code repository is: Simpler is better!

## Getting Started <a name = "getting_started"></a>


### Installing

requestments: todo

```
pip install -r requestments.txt
```



## Usage <a name = "usage"></a>

```
python main.py --data_path [] --model_name [] --model_saved_path [] --type_obj [] --train_data_path [] --test_data_path []
```

## 参数介绍

## 文件目录介绍

## 开发日志
1. 优化读取文件（增加用户指定训练集和测试集位置）
2. 区分DL和ML模型的构建
3. DL模型的参数文件撰写
4. 处理DL的数据集兼容整体的DATAloader通用方法
5. plt.show阻塞问题，换成显示1S，然后保存在当前目录下
6. 深度学习中数据的处理（转换id，构建词表）
7. dataset构建
8. lstm代码完成
9. 模型权重初始化代码
10. 模型训练代码
11. 模型评估代码
12. 早停trick
13. 参数优化
14. lstm 输出bug
15. dl_model test的代码
16. dl_model predict的代码
17. main函数的ML和DL的test和predict代码
18. 词向量的载入（todo
19. 深度学习，单个数据集输入的话，如何自动下采样和划分数据集（todo
20. 其他深度网络（todo
    1.  lstm
    2.  cnn
    3.  
21. 英文文本的分类（主要是分词在哪里方便修改）todo