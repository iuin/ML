import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 随机生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.5 + 1.2

# 定义两个变量, 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))

# 梯度下降法, 0.2的学习率
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 最小化代价函数, 这个是训练目的,使得 y_data与y之间的差值最小，使得定义的线性模型与样本模型基本一致
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k, b]))



