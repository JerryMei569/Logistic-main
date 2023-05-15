# coding:utf-8
import pandas as pd
from models.logistic import LogisticRegression
import numpy as np

# 读取csv文件
df = pd.read_csv('data/data.csv', encoding='gbk')

# 特征矩阵
X = df.loc[:, df.columns != '标签']

# 标签数组
y = df['标签']

# 初始化模型
logreg = LogisticRegression()

# 训练模型
logreg.fit(X, y)

# 保存权重到文件
np.save('weights.npy', logreg.weights)