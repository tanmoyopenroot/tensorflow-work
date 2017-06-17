import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sys
#%matplotlib inline

print("Python Version " + sys.version)
print("Tensorflow Version " + tf.VERSION)
print("Pandas Version " + pd.__version__)

df = pd.DataFrame({
		'a' : [2, 4, 6, 8],
		'b' : [2, 2, 2, 2]
	})

print(df)

a = tf.placeholder(tf.int32, name="var_a")
b = tf.placeholder(tf.int32, name="var_b")

mul = tf.multiply(a, b)

with tf.Session() as sess:
	print(sess.run(mul, feed_dict = {a : df['a'].tolist(), b : df['b'].tolist()}))


x = np.array([[1, 2, 3, 4, 5]])
y = np.array([[1, 1, 2, 3, 1]])

train_x = np.random.rand(100).astype(np.float32)
train_y = 0.1 * train_x + 0.3
#df.plot.scatter(x = 'a', y = 'b', figsize = (15,5));
plt.plot(train_x, train_y)
plt.show()

with tf.Session() as sess:
	print(sess.run(mul, feed_dict = {a: x, b: y}))


