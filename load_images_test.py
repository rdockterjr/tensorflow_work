import cv2

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50
NUMBER_OF_CHANNELS = 3
BATCH_SIZE = 30
NUM_EPOCHS = 1000

#To get images from google
#https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf


#helpful links
#https://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
#https://stackoverflow.com/questions/41483772/using-queues-in-tensorflow-to-load-images-and-labels-from-text-file
#http://web.stanford.edu/class/cs20si/lectures/slides_09.pdf
def create_photo_and_label_batches(dirlist):
    #get number of directories
    n_dirs = len(dirlist)

    #preallocate lists
    labels_list = []
    filenames_list = []
    for ff in range(0,n_dirs):
        #Get all files in image directories
        path_temp = os.getcwd() + dirlist[ff]
        allfiles = glob.glob(path_temp)
        filenames_list.extend(allfiles)

        #get the number of labels for this directory
        n_labels = len(allfiles)
        print "number of files:",n_labels
        
        
        # add labels to list
        for ii in range(0,n_labels):
            labels_list.append(ff)


    print "total  files:",len(filenames_list)
    
    # convert the lists to tensors
    filenames = tf.convert_to_tensor(filenames_list, dtype=tf.string)
    labels = tf.convert_to_tensor(labels_list, dtype=tf.int32)
    labels_one_hot = tf.one_hot(labels, n_dirs, on_value=1.0, off_value=0.0)

    # create queue with filenames and labels
    file_names_queue, labels_queue = tf.train.slice_input_producer([filenames, labels_one_hot], shuffle=True)

    # convert filenames of photos to input vectors
    file_contents = tf.read_file(file_names_queue)  # convert filenames to content
    photos_queue = tf.image.decode_jpeg(file_contents, channels=NUMBER_OF_CHANNELS)
    photos_queue.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUMBER_OF_CHANNELS])
    photos_queue = tf.to_float(photos_queue)  # convert uint8 to float32
    photos_queue = tf.reshape(photos_queue, [-1]) # flatten the tensor

    print type(photos_queue)
    # slice the data into mini batches
    return tf.train.batch([photos_queue, labels_queue], batch_size=BATCH_SIZE)


#get our images and labels
alldirs = []
alldirs.append('/animals/bear/*.jpg')
alldirs.append('/animals/flamingo/*.jpg')
training_photo_batch, training_label_batch = create_photo_and_label_batches(alldirs)

#get sizes
(n_train, n_features) = np.shape(training_photo_batch)
(n_train2, n_outputs) = np.shape(training_label_batch)
print(n_train)
print(n_features)
print(n_outputs)



