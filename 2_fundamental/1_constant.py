import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一个常量OP m1
m1 = tf.constant([[3, 3]])
# 创建一个常量OP m2
m2 = tf.constant([[2], [4]])

# 创建一个矩阵乘法的OP, 将m1和m2传入
product = tf.matmul(m1, m2)

with tf.Session() as sess:
    # run 方法触发了3个OP
    result = sess.run(product)

# 结果等于3*2+3*4
print(result)

