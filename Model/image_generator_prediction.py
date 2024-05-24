import tensorflow as tf
import math
import os
import random
import numpy as np
from Preprocessing import Preprocessing
from pathlib import Path



class ImageMaskGenerator(tf.keras.utils.Sequence):

    def __init__(self, images_folder, masks_folder, batch_size, nb_classes, validation=False, split=0.2, seed=42, augmentation=False, resize=False, size=(32,32,12), train=True):
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.seed = seed
        self.images_list = []
        self.masks_list = []
        self.augmentation = augmentation
        self.resize=resize
        self.size=size



        isExist = os.path.exists(images_folder)
        print(isExist)

        for root, _, files in os.walk(images_folder):
            files.sort()
            for file in files:
                self.images_list.append(os.path.join(root, file))
    



        for root, _, files in os.walk(masks_folder):
            # Here we sort to have the folder in alphabetical order
            files.sort()
            for file in files:
                self.masks_list.append(os.path.join(root, file))

        data_len = len(self.images_list)
        split_index = int(split * data_len)
        if train:
            random.Random(self.seed).shuffle(self.images_list)
            random.Random(self.seed).shuffle(self.masks_list)

        if validation:
            self.images_list = self.images_list[:split_index]
            self.masks_list = self.masks_list[:split_index]
        else:
            self.images_list = self.images_list[split_index:]
            self.masks_list = self.masks_list[split_index:]



    def __len__(self):
        return math.ceil(len(self.images_list) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.images_list[idx * self.batch_size:(idx + 1) *
                                                         self.batch_size]
        batch_y = self.masks_list[idx * self.batch_size:(idx + 1) *
                                                        self.batch_size]
        # print(batch_x)
        images=[]
        fileList=[]
        fileMask=[]

        for file in batch_x:
            a = np.load(file)
            p=Path(file).name
            fileList.append(p)
            if self.resize:
                prep=Preprocessing()
                a=prep.resize_with_padding(a, size=self.size)
            images.append(a)



            
        images_batch=np.array(images).astype('float')

        masks_raw=[]
        for file in batch_y:
            m=np.load(file)
            p=Path(file).name
            fileMask.append(p)
            if self.resize:
                prep=Preprocessing()
                m=prep.resize_with_padding(m, size=(self.size[0],self.size[1],1))
            masks_raw.append(m)
        print(len(masks_raw))
        masks_batch=np.array(masks_raw).astype('float')
        print(masks_batch.shape)
        return images_batch, masks_batch, fileList, fileMask

