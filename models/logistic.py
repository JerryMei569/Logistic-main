# coding:utf-8
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=True):
        """
        初始化逻辑回归模型
        :param lr: 学习率
        :param num_iter: 迭代次数
        :param fit_intercept: 是否拟合截距,True则有截距
        :param verbose: 是否打印迭代日志
        """
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.intercept_ = 0.0

    def __add_intercept(self, X):
        """
        添加截距,在X的列维度添加一列1
        :param X: 特征矩阵
        :return:添加了截距的特征矩阵X_
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        """
        sigmoid激活函数
        :param z: 激活前的值
        :return: 经过sigmoid激活后的值
        """
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        """
        二分类的交叉熵损失函数
        :param h: sigmoid后的预测概率,scalar or ndarray
        :param y: 真实标签,scalar or ndarray
        :return: 交叉熵损失,scalar
        """
        return -(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        """
        训练逻辑回归模型
        :param X: 特征矩阵
        :param y: 标签
        """
        X = np.array(X)
        y = np.array(y)
        print('y:', y)

        # 添加截距
        if self.fit_intercept:
            X = self.__add_intercept(X)
        print('X:', X)

            # weights初始化为0
        self.weights = np.zeros(X.shape[1])
        print(self.weights)

        # 迭代训练
        for i in range(self.num_iter):
            # 计算预测值

            z = np.dot(X, self.weights)

            h = self.__sigmoid(z)

                # 计算梯度和intercept
            dw = 0
            for j in range(len(X)):
                dw += (h[j] - y[j]) * X[j] # h[i]和X[i]都是标量,可以相乘
            dw += np.dot((h - y), X)
            #db = (h - y)
            dw = dw/len(X)


            # 更新权重
            self.weights -= self.lr * dw

            #if self.fit_intercept:
                #self.intercept_ -= self.lr * db

            # 参数
            if i == 99999:
                print('z:', z)
                print('h:', h)
                print('dw:', dw)
                print('weights:',self.weights)
                #print('db:', db)

            # 日志输出
            if self.verbose and i % 10000 == 0:
                print(f'损失值: {np.mean(self.__loss(h, y))}')

    def predict_prob(self, X):
        """
        预测概率
        :param X: 特征矩阵
        :return: 预测概率
        """
        z = 0
        z = np.dot(X, self.weights)
        print('X', X)
        print('weights:', self.weights)
        return self.__sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        预测类别
        :param X: 特征矩阵
        :param threshold: 分类阈值
        :return: 预测类别
        """
        if self.fit_intercept:
            X = self.__add_intercept(X)

        prob = self.predict_prob(X)
        return (prob >= threshold).astype('int')