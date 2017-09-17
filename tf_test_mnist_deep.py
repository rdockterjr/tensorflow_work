from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import argparse
import sys
from sys import exit

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from random import randint


print("Tensorflow test script, mnist deep learning")

### Using convolutional neural networks to classify mnist 
# CNN: the neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner

#get mnist data set
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#get data size
#vector inputs should have 55000 samples with 784 dimensions each
#vector outputs should have 55000 samples with 10 dimensions each
(n_train, n_features) = np.shape(mnist.train.images)
(n_train2, n_outputs) = np.shape(mnist.train.labels)
print(n_train)
print(n_features)
print(n_outputs)

print(type(mnist.train.images))

# Mnist data set consists of handwritten images of digits
# Images are 28x28 pixels, compressed into 1D array (size=784)
# mnist.train.images is size=55000x784
# mnist.train.labels are one-hot vectors [0,0,0,1,0,0,0,0,0,0] = 3
# mnist.train.labelsimages is size=55000x10

#NN Architecture
# Convolution -> pooling -> convolution -> pooling -> fully connected -> dropout -> readout(softmax)

#params for training
alpha = 1e-4 #learning rate
num_epochs = 20000 #training epochs
batch_size = 50 #number of samples per epochs


#placeholders for inputs and estimated outputs
x = tf.placeholder(tf.float32, [None, n_features])
y_in = tf.placeholder(tf.float32, [None, n_outputs])



#helper functions to initialize a lot of weights and bias variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#helper functions for convolution and pooling
#boundaries and stride size functions
#pooling helps downsample our 3D NN architecture into fewer nodes
def conv2d(x, W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')


# First we reshape our input vector back to a 28x28 image with 1 color channel
x_image = tf.reshape(x, [-1,28,28,1])

# first convolutional layer
# convolution followed by max pooling
# compute 32 features from a 5x5 window in the image
# Max pool reduces image to 14x144
W_conv1 = weight_variable([5,5,1,32]) #[input1,input2,input3,output]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  #relu=max(0,x)
h_pool1 = max_pool_2x2(h_conv1)


# second convolutional layer
# convolution followed by a max pooling
# input is the output of layer 1, 14x14 inputs
# compute 64 features for each 5x5 convolution window
# Max pool reduces image to 7x7
W_conv2 = weight_variable([5,5,32,64]) #[input1,input2,input3,output]
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  #relu=max(0,x)
h_pool2 = max_pool_2x2(h_conv2)


# Densely connected layer
# Input is 7x7 image from layer 2
# Now we use a fully connected layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout layer to reduce overfitting
# set keep_prob: 0.0 to ignore this
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
# basic softmax layer to get classification vector
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#classifier
classifier = tf.argmax(y_conv, 1)

#define our cross entropy loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in, logits=y_conv))
#create our training object
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
#for getting accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_in,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#session
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
for i in range(num_epochs):
    batch = mnist.train.next_batch(batch_size)
    #if we are printing this epoch
    print(type(batch[0]))
    print(batch[0].shape)
    exit(0)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_in: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy: %g' % (i, train_accuracy))
        if train_accuracy == 1:
            break
        
    #otherwise just train
    train_step.run(feed_dict={x: batch[0], y_in: batch[1], keep_prob: 0.5})
    
#if we are done print out our test accuracy
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_in: mnist.test.labels, keep_prob: 1.0})                                 
print('test accuracy %g' % test_accuracy)

#1: Using our model to classify a random MNIST image from the original test set:
num = randint(0, mnist.test.images.shape[0])
img = mnist.test.images[num]


#get images for viewing
mat_image = img.reshape(28, 28)
plt.imshow(mat_image, cmap=plt.cm.binary)
plt.show()

#test classify
class_est = classifier.eval(feed_dict={x: [img], y_in: mnist.test.labels, keep_prob: 1.0})
print('Neural Network predicted',class_est[0])
print('Real label is:', np.argmax(mnist.test.labels[num]))

