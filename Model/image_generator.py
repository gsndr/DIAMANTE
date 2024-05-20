import tensorflow as tf
import math
import os
import random
import numpy as np
import cv2
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
        print(images_folder)
        
        
        
        
      

        for root, _, files in os.walk(images_folder):
            files.sort()
            # Here we sort to have the folder in alphabetical order
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

    def on_epoch_end(self):
        random.Random(self.seed).shuffle(self.images_list)
        random.Random(self.seed).shuffle(self.masks_list)

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
            #a=a/10000 #change this
            #a = (a - np.min(a)) / (np.max(a) - np.min(a))
         
            if self.resize:
                prep=Preprocessing()
                a=prep.resize_with_padding(a, size=self.size)
                #print(a.shape)
               
            images.append(a)
        images_batch=np.array(images).astype('float')
        #print('images')
        #print(images_batch.shape)
       
        
        
       
        #images_batch = np.array([np.load(file) for file in batch_x]).astype('float')
        masks_raw=[]
        for file in batch_y:
            m=np.load(file)
            if self.resize:
                prep=Preprocessing()
                m=prep.resize_with_padding(m, size=(self.size[0],self.size[1],1))
            masks_raw.append(m)
        masks_batch= np.array(masks_raw).astype('float')
        #print('masks')
        #print(masks_batch.shape)
        
       






        # print(images_batch.shape)

        #masks_raw = np.array([np.load(file) for file in batch_y])
        #masks_batch = np.array([mask for mask in masks_raw]).astype('float')
        if self.augmentation:
            import albumentations as A
            aug = A.HorizontalFlip(p=1)

            aug2 = A.VerticalFlip(p=1)
            aug3 = A.RandomRotate90(p=1)
            aug4 = A.Transpose(p=1)
            aug5 = A.GridDistortion(p=1)
            #aug6 = A.GaussNoise(p=1)


            #aug8=A.Superpixels(p=1)

            for i in range(0, len(masks_batch)):
                img = images_batch[i]
                mask = masks_batch[i]
                # print('shape')
                # print(images_batch.shape)
                # print(masks_batch.shape)
                augmented = aug(image=img, mask=mask)
                imgA = np.expand_dims(augmented['image'], axis=0)
                maskA = np.expand_dims(augmented['mask'], axis=0)
                images_batch = np.append(images_batch, imgA, axis=0)
                masks_batch = np.append(masks_batch, maskA, axis=0)

                augmented = aug2(image=img, mask=mask)
                imgA = np.expand_dims(augmented['image'], axis=0)
                maskA = np.expand_dims(augmented['mask'], axis=0)
                images_batch = np.append(images_batch, imgA, axis=0)
                masks_batch = np.append(masks_batch, maskA, axis=0)
                augmented = aug3(image=img, mask=mask)
                imgA = np.expand_dims(augmented['image'], axis=0)
                maskA = np.expand_dims(augmented['mask'], axis=0)
                images_batch = np.append(images_batch, imgA, axis=0)
                masks_batch = np.append(masks_batch, maskA, axis=0)
                augmented = aug4(image=img, mask=mask)
                imgA = np.expand_dims(augmented['image'], axis=0)
                maskA = np.expand_dims(augmented['mask'], axis=0)
                images_batch = np.append(images_batch, imgA, axis=0)
                masks_batch = np.append(masks_batch, maskA, axis=0)
                augmented = aug5(image=img, mask=mask)
                imgA = np.expand_dims(augmented['image'], axis=0)
                maskA = np.expand_dims(augmented['mask'], axis=0)
                images_batch = np.append(images_batch, imgA, axis=0)
                masks_batch = np.append(masks_batch, maskA, axis=0)
                '''
                augmented = aug6(image=img, mask=mask)
                imgA = np.expand_dims(augmented['image'], axis=0)
                maskA = np.expand_dims(augmented['mask'], axis=0)
                images_batch = np.append(images_batch, imgA, axis=0)
                masks_batch = np.append(masks_batch, maskA, axis=0)
                '''
  


            #print(images_batch.shape)
            #print(masks_batch.shape)
        return images_batch, masks_batch

