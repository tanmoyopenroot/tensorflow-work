import tensorflow as tf

a = tf.placeholder(tf.float32, name='var_a')
b = tf.placeholder(tf.float32, name='var_b')

c = tf.multiply(a, b)

with tf.Session() as session:
	print(session.run(c, feed_dict = {a:[7.0], b:[8.0]}))

