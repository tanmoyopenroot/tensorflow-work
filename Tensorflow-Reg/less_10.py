import tensorflow as tf
import numpy as np
import sys

pool = np.random.rand(1000, 1).astype(np.float32)
np.random.shuffle(pool)

sample = (int)(1000 * 0.15)

#Test Data
test_x = pool[0 : sample]
test_y = 2.0 * test_x**4 + 5

#Valid Data
valid_x = pool[sample : sample * 2]
valid_y = 2.0 * valid_x**4 + 5

#Train Data
train_x = pool[sample * 2 : ]
train_y = 2.0 * valid_x**4 + 5

def addLayer(x, in_size, out_size, act_func = None):
	W = tf.Variable(tf.truncated_normal([in_size, out_size], mean = 0.1, stddev = 0.1))
	b = tf.Variable(tf.truncated_normal([out_size], mean = 0.1, stddev = 0.1))
	pred = tf.add(tf.matmul(x, W), b)
	if act_func is None:
		output = pred
	else:
		output = act_func(pred)
	return output

#PlaceHolders
x = tf.placeholder(tf.float32, shape = [None, 1], name = "x")
y = tf.placeholder(tf.float32, shape = [None, 1], name = "y")


#Create Hidden Layer
hidden_size  = 100
h1 = addLayer(x, 1, hidden_size, tf.nn.relu)
h2 = addLayer(h1, hidden_size, hidden_size, tf.nn.relu)

#Display Shape
print("Hidden Layer 1: ", h1.get_shape())
print("Hidden Layer 2: ", h2.get_shape())


#Create Output Layer
pred = addLayer(h2, hidden_size, 1)
#Display Shape
print("Output Layer: ", pred.get_shape())

#Create Loss
loss = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.train.GradientDescentOptimizer(0.002)
train = optimizer.minimize(loss)

#Accuracy
correct_pred = tf.equal(tf.round(pred), tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Overfitting Variable
prev_valid_acc = 0.0
prev_impov_step = 0
req_impov_step = 1500

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for step in range(6000):
		train_data = {x : train_x, y : train_y}
		sess.run(train, feed_dict = train_data)

                train_loss = sess.run(loss, feed_dict = train_data)
                print("Training Loss: ", train_loss)
		
		if step % 100 == 0:
	                valid_data = {x : valid_x, y : valid_y}
			valid_acc = sess.run(accuracy, feed_dict = valid_data)
			if valid_acc  > prev_valid_acc:
				print("Improvement Found: Previous Accuracy: ", prev_valid_acc, " Current Accuracy: ", valid_acc)
				prev_valid_acc = valid_acc
				prev_impov_step = step
			if step - prev_impov_step > req_impov_step:
				print("No Impovement Found")
				break

	train_acc = sess.run(accuracy, feed_dict = train_data)
	print("Training Accuracy: ", train_data)
	test_data = {x : test_x, y : test_y}
	test_acc = sess.run(accuracy, feed_dict = test_data)	
	print("Test Accuracy: ", test_acc)
