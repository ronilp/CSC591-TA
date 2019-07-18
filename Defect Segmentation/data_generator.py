import os
import cv2
import keras
import numpy as np
from skimage.transform import resize


# This is generator class to process data in batches and send them for training
class Surface_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, test=False):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    # return the total number of batches you have i.e., total_files/batch_size
    def __len__(self):
        # YOUR CODE HERE
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    # this function is called for every mini-batch to get the images/masks for that mini-batch
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        image_arr = []
        mask_arr = []
        # Open a batch of images and their corresponding masks using cv2.imread
        # resize them to 512x512x1 and return an np.array of images and masks
        # YOUR CODE HERE
        for file_name in batch_x:
            if not os.path.isfile(file_name):
                print(file_name)

            img = resize(cv2.imread(file_name, 0), (512, 512, 1), mode='constant')
            image_arr.append(img)

        for file_name in batch_y:
            if not os.path.isfile(file_name):
                print(file_name)

            img = resize(cv2.imread(file_name, 0), (512, 512, 1), mode='constant')
            mask_arr.append(img)

        return np.array(image_arr).astype(np.float32), np.array(mask_arr).astype(np.float32)

    # for testing we need to get the list of all true masks
    # this function should return all the labels in the dataset set
    # we will call this function only for the "Test" dataset
    def get_all_masks(self):
        mask_arr = []

        # YOUR CODE HERE
        for file_name in self.labels:
            img = resize(cv2.imread(file_name, 0), (512, 512, 1), mode='constant')
            mask_arr.append(img)

        return np.array(mask_arr).astype(np.float32)

