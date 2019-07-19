#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: insurance.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

__author__ = 'lebaishi'

data = pd.read_csv('./insurance.csv')
print(type(data))
print(data.head())
print(data.tail())
# describe做简单的统计摘要
print(data.describe())

# 采样要均匀
data_count = data['age'].value_counts()
print(data_count)
data_count[:10].plot(kind='bar')
plt.show()
#plt.savefig('./temp')

print(data.corr())

reg = LinearRegression()
x = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']
# python3.6 报错 sklearn ValueError: could not convert string to float: 'northwest'，加入一下几行解决
x = x.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
x.fillna(0, inplace=True)
y.fillna(0, inplace=True)

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(x)

reg.fit(X_poly, y)
print(reg.coef_)
print(reg.intercept_)

y_predict = reg.predict(X_poly)

plt.plot(x['age'], y, 'b.')
plt.plot(X_poly[:, 0], y_predict, 'r.')
plt.show()
