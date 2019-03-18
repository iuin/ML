import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fetch 指在一个会话中可以同时运行多个OP，传递一个OP数组给会话
# feed
# 创建一个变量input1
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)

# 乘法OP
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    # 这个就是fetch操作
    print(sess.run([mul, add]))


# feed
# 创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # 在运行的时候再传入值, feed的数据以字典形式传入， 这里就不需要再初始化变量的
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
