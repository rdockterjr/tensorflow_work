import tensorflow as tf

#import mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#placeholder for inputs
x = tf.placeholder(tf.float32, [None, 784])

#variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implement model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#true class label output placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#training step
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

#start the session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#actual training steps
for _ in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#test output
print(batch_ys[0])
print(y[0])

#test the prediciton
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
