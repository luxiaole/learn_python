import numpy as np
"""
SGD：随机梯度下降法
"""
__author__ = 'yasaka'
#批量梯度下降法
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
# print(X_b)
#设置学习率
learning_rate = 0.1
#设置迭代次数
n_iterations = 100000
m = 100

# 1，初始化theta，w0...wn
theta = np.random.randn(2, 1)
count = 0

# 4，不会设置阈值，之间设置超参数，迭代次数，迭代次数到了，我们就认为收敛了
for iteration in range(n_iterations):
    count += 1
    # 2，接着求梯度gradient
    gradients = 1/m * X_b.T.dot(X_b.dot(theta)-y)
    # 3，应用公式调整theta值，theta_t + 1 = theta_t - grad * learning_rate
    theta = theta - learning_rate * gradients

print(count)
print(theta)