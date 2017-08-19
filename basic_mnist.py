# Import all our libraries

import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

#training parameters
alpha = 0.5 #learning rate
iterations = 1000
samples = 100

#import the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train)

#get data size
nimages,simages = mnist.train.images.shape #55000, 784
nlabels,slabels = mnist.train.labels.shape #55000, 10
print("Number of dimensions: ",simages,slabels)
print("Number of training samples: ", nimages)

#placeholders for inputs
x = tf.placeholder(tf.float32, [None, simages])
vec_label = tf.placeholder(tf.float32, [None, slabels])

#variables for outputs
W = tf.Variable(tf.zeros([simages,slabels]))
b = tf.Variable(tf.zeros([slabels]))

#softmax model
y = tf.nn.softmax(tf.matmul(x, W) + b)
classify = tf.arg_max(y,1)
labeler = tf.arg_max(vec_label,1)

#define our cross entropy cost function sum(yhat*log(y)
y_hat = tf.placeholder(tf.float32, [None, slabels]) #input label
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_hat, logits=y))

#now define our gradient descent optimizer:
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

#launch the tensorflow session
sess = tf.InteractiveSession()

#initialize variables
tf.global_variables_initializer().run()

#run our training loop
for _ in range(iterations):
    batch_x, batch_y = mnist.train.next_batch(samples)
    sess.run(train_step, feed_dict={x: batch_x, y_hat: batch_y})


#Check how good our classification is
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sample_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_hat: mnist.test.labels})
print("Accuracy=",sample_acc)

#sample id
sid = 15
test_label = mnist.test.labels[sid].reshape(1,slabels)
test_image = mnist.test.images[sid].reshape(1,simages)

#get estimated label and true label
label_est = sess.run(classify, feed_dict={x: test_image})
label_true = sess.run(labeler, feed_dict={vec_label: test_label})
print("estimated: ", label_est)
print("true: ", label_true)

#show image
display_image = mnist.test.images[sid].reshape(28,28)
plt.imshow(display_image)
plt.show()

