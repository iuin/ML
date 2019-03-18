import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一个变量v
v = tf.Variable([1, 2])
# 创建一个常量c
c = tf.constant([3, 3])

# 增加一个减法OP
sub = tf.subtract(v, c)

# 增加一个加法OP
add = tf.add(v, sub)

# 初始化变量操作
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 下面这一句不能省略，需要执行
    sess.run(init)
    # 打印结果
    print(sess.run(sub))  # [-2 -1]
    print(sess.run(add))  # [-1  1]


# 定义一个变量state, name为count
state = tf.Variable(0, name="count")

# 将state的值增加1
new_value = tf.add(state, 1)

# 赋值操作，将new_value的值赋给state
update = tf.assign(state, new_value)
# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        print(sess.run(update))
