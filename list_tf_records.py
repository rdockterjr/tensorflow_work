#helper funcitons to list all tf record files in a given directory

import os  # handle system path and filenames
import tensorflow as tf  # import tensorflow as usual

# define a function to list tfrecord files, one list for train, one list for validate
def list_tfrecord_file(file_list,data_dir):
    tftrain_list = []
    tfvalidate_list = []
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(data_dir + '/' + file_list[i])		
        if current_file_abs_path.endswith(".tfrecord"):
            if "train" in current_file_abs_path:
                tftrain_list.append(current_file_abs_path)
                #print("Train File: %s " % file_list[i])
            elif "validation" in current_file_abs_path:
                tfvalidate_list.append(current_file_abs_path)
                #print("Validate File: %s " % file_list[i])
            else:
                pass
        else:
            pass
    return tftrain_list, tfvalidate_list

# Traverse current directory
def tfrecord_auto_traversal(data_dir, printdebug=True):
    current_folder_filename_list = os.listdir(data_dir) # Change this PATH to traverse other directories if you want.
    if current_folder_filename_list != None:
        if printdebug:
            print("%s files were found under current folder. " % len(current_folder_filename_list))
            print("Please be noted that only files end with '*.tfrecord' will be loaded!")
        tftrain_list, tfvalidate_list = list_tfrecord_file(current_folder_filename_list,data_dir)
        if printdebug:
            if len(tftrain_list) != 0:
                print("Training Files:")
                for list_index in xrange(len(tftrain_list)):
                    print(tftrain_list[list_index])
            else:
                print("Cannot find any training files, please check the path.")
            if len(tfvalidate_list) != 0:
                print("Validation Files:")
                for list_index in xrange(len(tfvalidate_list)):
                    print(tfvalidate_list[list_index])
            else:
                print("Cannot find any validation files, please check the path.")
    return tftrain_list, tfvalidate_list


def main():
    data_dir = './animal_tfrecord' #change to whatever folder you use for output directory
    tfrecord_list = tfrecord_auto_traversal(data_dir)

if __name__ == "__main__":
    main()
