# coding:utf-8
import numpy as np
from models.logistic import LogisticRegression

X_test = [[5,2,55],[56,7,4],[5,3,1],[4,13,19],[1,90,2],[4,12,8]]
X_test = np.array(X_test)

# 加载权重
weights = np.load('weights.npy')

# 初始化模型
logreg = LogisticRegression()

# 进行预测
logreg.weights = weights
y_pred = logreg.predict(X_test)

print('预测结果是：', y_pred)