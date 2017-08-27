#Deep learning with mnist
# Import all our libraries

import sys
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import numpy as np

#training parameters
alpha = 1e-4 #learning rate
iterations = 20000
samples = 100
pool_size = 2
kernel_size = 5
output_layer_1 = 32
output_layer_2 = 64

#helper functions
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, pool_size, pool_size, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#get data size
nimages,simages = mnist.train.images.shape #55000, 784 
nlabels,slabels = mnist.train.labels.shape #55000, 10
print("Number of dimensions: ",simages,slabels)
print("Number of training samples: ", nimages)


#NOW WE BUILD UP OUR LAYERS:

#FIRST LAYER:
# placeholder inputs
x = tf.placeholder(tf.float32, [None, simages])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_true = tf.placeholder(tf.float32, [None, slabels])

#weights and biases
W_conv1 = weight_variable([kernel_size, kernel_size, 1, output_layer_1])
b_conv1 = bias_variable([output_layer_1])

#output function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#SECOND LAYER:
#weights and biases
W_conv2 = weight_variable([kernel_size, kernel_size, output_layer_1, output_layer_2])
b_conv2 = bias_variable([output_layer_2])

#output function
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#DENSELY CONNECTED LAYER:
#weights and biases
W_fc1 = weight_variable([7 * 7 * output_layer_2, 1024])
b_fc1 = bias_variable([1024])

#output function
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*output_layer_2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#Dropout to reduce over fitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#Readout layer
W_fc2 = weight_variable([1024, slabels])
b_fc2 = bias_variable([slabels])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_conv))

#training funciton
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#start the session again
sess = tf.InteractiveSession()
print(device_lib.list_local_devices())


#initialize variables
sess.run(tf.global_variables_initializer())

#test runtime
t0 = time.time()

for i in range(iterations):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})

#test runtime
t1 = time.time()
total = t1-t0
print("Training Time (s): ", total)

print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1.0}))


