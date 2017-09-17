import cv2

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob
from random import randint

from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image

from list_tf_records import tfrecord_auto_traversal

RECORD_DIR = './animal_tfrecord' #change to whatever folder you use for output directory
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_CHANNELS = 3
BATCH_SIZE = 20 #number of images to use each epoch
NUM_EPOCHS = 1000 #total training iterations
IMAGE_NUMBER = 137 # Number of images in data set
CLASS_NUMBER = 2 #number of unique classes
alpha = 1e-4 # training rate

#To get images from google
#https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf

#This was taken from
#http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string)
        self.height = tf.Variable([], dtype = tf.int64)
        self.width = tf.Variable([], dtype = tf.int64)
        self.filename = tf.Variable([], dtype = tf.string)
        self.label = tf.Variable([], dtype = tf.float32)
        self.minwidth = tf.Variable([], dtype = tf.int32)


        
def read_and_decode(filename_queue):
    #loads up images from our tfrecord files and resizes the images, also converts our labels to one hot
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)

    #convert label to one hot, load_image_data.py just stores it as an integer
    rawlabel = tf.cast(features["image/class/label"], tf.int32) # label of the raw image
    onehotlabel = tf.one_hot(rawlabel,CLASS_NUMBER,on_value=1.0,off_value=0.0)

    #resize the image
    current_image_object = image_object()
    current_image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_HEIGHT, IMAGE_WIDTH) # cropped image with size 299x299

    #store the rest of the info
    current_image_object.height = features["image/height"] # height of the raw image
    current_image_object.width = features["image/width"] # width of the raw image
    current_image_object.filename = features["image/filename"] # filename of the raw image
    current_image_object.label = onehotlabel;
    
    return current_image_object

#return numpy array of images and labels
def next_batch(record_dir, batch_size, if_random = True, if_training = True):
    #make sure we dont ask for more batches than we have
    if batch_size > IMAGE_NUMBER:
        batch_size = IMAGE_NUMBER

    #traverse a folder for all tfrecord files
    trainfile_list,validatefile_list = tfrecord_auto_traversal(record_dir,True)

    file_list = trainfile_list
    if if_training != True:
        file_list = validatefile_list
        
    #Get all the tfrecord files from our directory
    filename_queue = tf.train.string_input_producer( file_list )

    #get all our images
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)
    
    label = image_object.label
    filename = image_object.filename

    if(if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(IMAGE_NUMBER * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = batch_size,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * batch_size,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = batch_size,
            num_threads = 1)
    return image_batch, label_batch, filename_batch


        
#We actually want to use our images in a model

#weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolution and pooling funcitons
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def cnn_model(x, keep_prob):
    #### First convolutional layer (image starts out at 100x100)
    #First two dimensions: patch size, 3rd dimension: number of input channels, 4th dimension: number of output channels. 
    out_channels_1 = 32
    patch_size_1 = 5
    W_conv1 = weight_variable([patch_size_1, patch_size_1, IMAGE_CHANNELS, out_channels_1])
    b_conv1 = bias_variable([out_channels_1])

    #convolving x input with the weights and relu:
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) #(image = 50x50 now)


    #### Second convolutional layer
    #First two dimensions: patch size, 3rd dimension: number of input channels, 4th dimension: number of output channels. 
    out_channels_2 = 64
    patch_size_2 = 5
    W_conv2 = weight_variable([patch_size_2, patch_size_2, out_channels_1, out_channels_2])
    b_conv2 = bias_variable([out_channels_2])

    #convolving output of layer 1 with the weights and relu:
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) #(image = 25x25 now)


    #### Densely Connected Layer
    #now the image has been reduced by half each step so 100->50->25 = 25x25 image
    conv_layers = 2
    out_channels_3 = 1024
    dense_width = IMAGE_WIDTH / (2 * conv_layers)
    dense_height= IMAGE_HEIGHT / (2 * conv_layers)

    #fully connected layer with 2014 neurons for entire image
    W_fc1 = weight_variable([dense_width * dense_height * out_channels_2, out_channels_3])
    b_fc1 = bias_variable([out_channels_3])

    #convolving the output
    h_pool2_flat = tf.reshape(h_pool2, [-1, dense_width * dense_height * out_channels_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    #### dropout to reduce over fitting
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    #### Output layer with one node for each class
    W_fc2 = weight_variable([out_channels_3, CLASS_NUMBER])
    b_fc2 = bias_variable([CLASS_NUMBER])
    
    #final output value
    y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_out



#function to call for training
#https://github.com/yeephycho/tensorflow_input_image_by_tfrecord/blob/master/src/flower_train_cnn.py
def model_train():
    #get batch from our own dataset
    image_batch, labels_batch, files_batch = next_batch('./animal_tfrecord',BATCH_SIZE,True,True)

    #placeholders for input features and labels
    x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, CLASS_NUMBER])
    keep_prob = tf.placeholder(tf.float32)

    #Use our cnn as our model output
    y_conv = cnn_model(x, keep_prob)

    #classifier
    classifier = tf.to_int64(tf.argmax(y_conv, 1))

    #### Define our training and cost function
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
    
    #for checking accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #for saving our model and so we can keep going with a training later on
    saver = tf.train.Saver()

    #start our session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/home/ubuntu/python_ws/training_store/checkpoint-train.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)
        
        #start the training
        print("Starting Training Epochs")
        for i in range(NUM_EPOCHS):
            #get our images
            image_out, label_out = sess.run([image_batch, labels_batch])

            #run training step
            _, loss_out = sess.run([train_step, cross_entropy], feed_dict={x: image_out, y_: label_out, keep_prob: 0.5 })
            
            #if we are printing this epoch
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: image_out, y_: label_out, keep_prob: 1.0})
                print('Epoch %d, training accuracy: %g' % (i, train_accuracy))

            #if we should save a checkpoint this epoch
            if i % 20 == 0 :
                saver.save(sess, "/home/ubuntu/python_ws/training_store/checkpoint-train.ckpt")        


    coord.request_stop()
    coord.join(threads)
    sess.close()


#function to call for testing
def model_eval():
    #get batch from our own dataset
    image_batch, labels_batch, files_batch = next_batch('./animal_tfrecord',BATCH_SIZE,False,False)

    #placeholders for input features and labels
    x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, CLASS_NUMBER])
    keep_prob = tf.placeholder(tf.float32)

    #Use our cnn as our model output
    y_conv = cnn_model(x, keep_prob)

    #classifier
    classifier = tf.to_int64(tf.argmax(y_conv, 1))
    reverse_onehot = tf.to_int64(tf.argmax(y_, 1))

    #### Define our training and cost function
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
    
    #for checking accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #for saving
    saver = tf.train.Saver()

    #setup our session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/home/ubuntu/python_ws/training_store/checkpoint-train.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        test_samples = 20
        accuracy_accu = 0
        num = randint(0, test_samples)

        for i in range(test_samples):
            #get out the next batch
            image_out, label_out = sess.run([image_batch, labels_batch])


            #test the classification
            accuracy_out, logits_out, label_true = sess.run([accuracy, classifier, reverse_onehot], feed_dict={x: image_out, y_: label_out, keep_prob: 1.0})
            accuracy_accu += accuracy_out

            #print out some info
            print(i)
            print("temp_acc: ")
            print(accuracy_out)
            print("label true: ")
            print(label_true)
            print("label est: ")
            print(logits_out)

            if(i == num):
                print("Sample Image")

                #1: Using our model to classify a random  image from the original test set:
                img = image_out[0]

                #get images for viewing
                plt.imshow(img)
                plt.show()

                #test classify
                print('True Label: ' % label_true[0])
                print('Est Label:' % logits_out[0])

        print("Accuracy: ")
        print(accuracy_accu / test_samples)

        coord.request_stop()
        coord.join(threads)
        sess.close()

#run the training
#model_train()
model_eval()
