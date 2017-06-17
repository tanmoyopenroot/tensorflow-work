import tensorflow as tf
import numpy as np
import sys
import datetime

#Generate Samples
pool = np.random.rand(1000,1).astype(np.float32)
np.random.shuffle(pool)

sample = (int)(1000 * 0.15)

#15% test
test_x = pool[0 : sample]
test_y = 2.0 * test_x**2 + 3.0 * test_x + 5

#15% validation
valid_x = pool[sample : sample * 2]
valid_y = 2.0 * valid_x**2 + 3.0 * valid_x + 5

#70% training
train_x = pool[sample * 2 : ]
train_y = 2.0 * train_x**2 + 3.0 * train_x + 5


def addLayer(x, in_size, out_size, act_func = None):
	W = tf.Variable(tf.truncated_normal([in_size, out_size], mean = 0.1, stddev = 0.1))
	b = tf.Variable(tf.truncated_normal([out_size], mean = 0.1, stddev = 0.1))
	
	pred = tf.add(tf.matmul(x, W), b)
	
	if act_func is None:
		output = pred
	else:
		output = act_func(pred)
	return output

x = tf.placeholder(tf.float32, shape = [None, 1], name = "x")
y = tf.placeholder(tf.float32, shape = [None ,1], name = "y")
keep_drop = tf.placeholder(tf.float32)

#Create Hidden Layers
init_size = 100
h1 = addLayer(x, 1, init_size, tf.nn.relu)
h2 = addLayer(h1, init_size, init_size, tf.nn.relu)
h3 = addLayer(h2, init_size, init_size, tf.nn.relu)

print("Hidden Layer 1: ",h1.get_shape())
print("Hidden Layer 2: ",h2.get_shape())
print("Hidden Layer 3: ",h3.get_shape())

#Creating Output Layer
pred = addLayer(h3, init_size, 1)

loss = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.train.GradientDescentOptimizer(0.8)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.round(pred), tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#Validation Init
prev_valid_acc = 0.0
prev_valid_step = 0
step_acc_improv = 1000


init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print("Staring Training")
	for step in range(6000):
		train_data = {x : train_x, y : train_y}
		sess.run(train, feed_dict = train_data)
		
		#Generating Loss
		train_loss = sess.run(loss, feed_dict = train_data)
		
		#Getting Current Training Accuracy
		train_acc = sess.run(accuracy, feed_dict = train_data)
		print("Current Train Accuracy: ", train_acc , "Train Loss: ", train_loss, "Step: ", step)

		if step % 100 == 0:		
			#Validation Testing
			valid_data = {x: valid_x, y : valid_y}
			valid_acc = sess.run(accuracy, feed_dict = valid_data)
			if valid_acc > prev_valid_acc:
                                print("Improvement Found At Step: ", step)
				print("Improvement Difference: ", valid_acc - prev_valid_acc)
				prev_valid_acc = valid_acc
				prev_valid_step = step
			if step - prev_valid_step > step_acc_improv:
				print("No Improvement Found After ", step_acc_improv)
				break

	print("Training Done")
	train_acc = sess.run(accuracy, feed_dict = train_data)
	print("Train Accuracy :", train_acc)

	test_data = {x : test_x, y : test_y}
	test_acc = sess.run(accuracy, feed_dict = test_data)
	print("Test Accuracy : ", test_acc)

