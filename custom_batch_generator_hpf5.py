import numpy as np
import keras
import h5py
import os
import cv2

class myDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path_images_h5file=None, path_masks_h5file=None, batch_size=8, img_dim=(256,256,3), mask_dim=(256,256),
                 shuffle=True):
        'Initialization'
        self.img_dim = img_dim
        self.mask_dim = mask_dim
        self.batch_size = batch_size 
        self.shuffle = shuffle

        self.hfImg = h5py.File( path_images_h5file, 'r')
        if self.hfImg is None:
            raise Execption("Image data file read error!")

        self.hfMask = h5py.File( path_masks_h5file, 'r')
        if self.hfMask is None:
            raise Execption("Mask data file read error!")

        self.sampleCount = len(self.hfImg.keys())

        self.indexes = np.arange(self.sampleCount)
        np.random.shuffle(self.indexes) 
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.sampleCount / self.batch_size))

    def on_epoch_end(self):
        'shuffle indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        sampleIDs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = np.empty((self.batch_size, *self.img_dim))
        M = np.empty((self.batch_size, *self.mask_dim))

        i = 0
        for ID in sampleIDs:
            X[i] = np.array(self.hfImg[str(ID)])
            M[i] = np.array(self.hfMask[str(ID)])
            i += 1
        return X, M


