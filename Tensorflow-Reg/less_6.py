import tensorflow as tf
import numpy as np
import sys
import math

train_x = np.random.rand(100).astype(np.float32)
train_y = 0.1 * train_x + 0.3

test_x = np.random.rand(100).astype(np.float32)
test_y = 0.1 * train_x + 0.3

W = tf.Variable(np.random.rand(), name = "W")
b = tf.Variable(np.random.rand(), name = "b")

x = tf.placeholder(tf.float32, name = "x")
y = tf.placeholder(tf.float32, name = "y")

pred = tf.add(tf.multiply(W, x) , b)
loss = tf.reduce_mean(tf.square(pred - train_y))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.round(pred), tf.round(train_y))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(6000):
		train_data = {x : train_x, y : train_y}
		sess.run(train, feed_dict = train_data)
		print("W = ", sess.run(W), "b = ", sess.run(b))
	print("Training Completed", "W = ", sess.run(W), "b = ", sess.run(b))	
	
	print("Accuracy: " , sess.run(accuracy, feed_dict = {x : train_x, y : train_y}))

	test_results = sess.run(pred, feed_dict = {x : test_x})
	print(test_results)



