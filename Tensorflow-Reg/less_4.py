import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_x = np.random.rand(100).astype(np.float32)
train_y = 0.1 * train_x + 0.3

print("Training Data Info")
df = pd.DataFrame({
		'x' : train_x,
		'y' : train_y
	})

df.head()
df.describe()

plt.plot(train_x, train_y)
plt.show()


#Creating Test Data
test_x = np.random.rand(100).astype(np.float32)

x = tf.placeholder(tf.float32, name = "x")
y = tf.placeholder(tf.float32, name = "y")

W = tf.Variable(np.random.rand())
b = tf.Variable(np.random.rand())

pred = tf.multiply(W, x) + b

loss = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.train.GradientDescentOptimizer(0.7)
train = optimizer.minimize(loss)

#Initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for step in range(200):
		train_data = {
			x : train_x,
			y : train_y
		}

		sess.run(train, feed_dict = train_data)

		if step > 100:
			print(step, sess.run(W), sess.run(b))
	
	print("Training Comleted:" , "W = ",  sess.run(W) , "b = " , sess.run(b))

	test_results = sess.run(pred, feed_dict = {x : test_x})

	plt.plot(test_x, test_results)
	plt.show()

	df_final = pd.DataFrame({
			'test_x' : test_x,
			'pred' : test_results
		})
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 5))
df.plot.scatter(x = 'x', y = 'y', ax = axes, color = 'red')
df_final.plot.scatter(x = 'test_x', y = 'pred', ax = axes, alpha = 0.3)

axes.set_title('target vs pred', fontsize = 20)
axes.set_ylabel('y', fontsize = 15)
axes.set_xlabel('x', fontsize = 15);	
