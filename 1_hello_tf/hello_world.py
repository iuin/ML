import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant('Hello Tensorflow!')

with tf.Session() as sess:
    result = sess.run(hello)

print(result)
