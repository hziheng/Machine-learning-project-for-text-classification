import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
 
classes = ['W','LS','SWS','REM']
confusion_matrix = np.array([(193,31,0,41),(87,1038,32,126),(17,337,862,1),(17,70,0,638)],dtype=np.int)#输入特征矩阵
proportion=[]
for i in confusion_matrix:
    for j in i:
        temp=j/(np.sum(i))
        proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
#print(proportion)
pshow=[]
for i in proportion:
    pt="%.2f%%" % (i * 100)
    pshow.append(pt)
proportion=np.array(proportion).reshape(4,4)  # reshape(列的长度，行的长度)
pshow=np.array(pshow).reshape(4,4)
#print(pshow)
config = {
    "font.family":'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
            # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
            # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes,fontsize=12)
plt.yticks(tick_marks, classes,fontsize=12)
 
thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(4)] for i in range(4)],(confusion_matrix.size,2))
for i, j in iters:
    if(i==j):
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12,color='white',weight=5)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12,color='white')
    else:
        plt.text(j, i-0.12, format(confusion_matrix[i, j]),va='center',ha='center',fontsize=12)   #显示对应的数字
        plt.text(j, i+0.12, pshow[i, j], va='center', ha='center', fontsize=12)
 
plt.ylabel('True label',fontsize=16)
plt.xlabel('Predict label',fontsize=16)
plt.tight_layout()
plt.show()
 