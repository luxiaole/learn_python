#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression

__author__ = 'lebaishi'
#通过sklearn使用LinearRegression求最优解

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
#intercept_:截距 coef_：参数
print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.array([[0], [2]])
print(lin_reg.predict(X_new))