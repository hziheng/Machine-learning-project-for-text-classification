# 机器学习代码模板（开箱即用）


- [机器学习代码模板（开箱即用）](#机器学习代码模板开箱即用)
  - [**关于** ](#关于-)
    - [1. **介绍**](#1-介绍)
    - [2. **目前已经涵盖的算法**](#2-目前已经涵盖的算法)
      - [2.1 常见的机器学习算法](#21-常见的机器学习算法)
      - [2.2 常见的深度学习算法](#22-常见的深度学习算法)
  - [**前提准备** ](#前提准备-)
    - [**环境安装**](#环境安装)
  - [**具体使用方法** ](#具体使用方法-)
    - [3. **参数介绍**](#3-参数介绍)
    - [3.1 **针对常见的机器学习算法**](#31-针对常见的机器学习算法)
    - [3.2 **针对深度神经网络算法**](#32-针对深度神经网络算法)
  - [文件目录介绍](#文件目录介绍)
  - [开发日志](#开发日志)


---

## **关于** <a name = "关于"></a>
### 1. **介绍**

> 这是一个包含多种机器学习算法的代码模板库，其主要用于NLP中文本分类的下游任务，包括二分类及多分类。使用者只需更改一些参数例如数据集地址，算法名称等，即可以使用其中的各种模型来进行文本分类（前提是数据集与我提供的数据集形式一致，具体可以看data/ 下我提供的数据集），各种算法的参数只在xx_config.py单个文件中提供，方便用户对神经网络模型进行调参。
### 2. **目前已经涵盖的算法**
#### 2.1 常见的机器学习算法

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- GBDT(Gradient Boosting Decision Tree)
- XGBoost
- Catboost
- SVM
- Bayes
- 待补充


#### 2.2 常见的深度学习算法

- TextCNN
- Bi-LSTM
- Transformer
- 待补充


---



## **前提准备** <a name = "前提准备"></a>

### **环境安装**

具体的相关库的版本见requestments.txt

- 使用命令安装

```
pip install -r requestments.txt
```



## **具体使用方法** <a name = "具体使用方法"></a>
### 3. **参数介绍**
***主程序：main.py，其中各个参数的含义如下：***

> *--data_path*: 一个完整的（未切分训练集测试集）的数据集路径
> 
> *--model_name*: 需要使用的算法名称，填写的简称见config.py中的ML_MODEL_NAME和DL_MODEL_NAME
> 
> *--model_saved_path*: 模型存储的路径
> 
> *--type_obj*: 程序的运行目的：train，test，predict三个选项
> 
> *--train_data_path*: 切分好的训练集路径
>
> *--test_data_path*: 切分好的测试集路径
> 
> *--dev_data_path*: 切分好的验证集路径
### 3.1 **针对常见的机器学习算法**

***终端命令如下：***
```
python main.py --data_path [] --model_name [] --model_saved_path [] --type_obj []
```
***示例***

```
python main.py --data_path ./data/processed_data.csv --model_name lg --type_obj train
```

解释：这里的train_data_path, test_data_path, dev_data_path都默认为空，ml的数据处理模块会自动按照7：3划分训练集和测试集，并且默认进行下采样，避免数据不平衡带来的不良影响，划分比例和是否下采样参数可在config.py自行修改，如果参数train_data_path, test_data_path被指定，则无需指定data_path, split_size, is_sample参数

***运行结果如下：***

![result.png](pic/result.png)

***结果图片展示：***

![result_ml.png](pic/pic_ml.png)

### 3.2 **针对深度神经网络算法**

***终端命令如下：***
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
18. 词向量的载入
19. 深度学习，单个数据集输入的话，如何自动下采样和划分数据集（todo
20. 其他深度网络（todo
    1.  lstm
    2.  cnn
    3.  transformer
21. 英文文本的分类（主要是分词在哪里方便修改）todo