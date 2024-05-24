import os
import numpy as np
from skimage import io
import tifffile as tiff


class Preprocessing():
    #def __init__(self):

    def resize_with_padding(self, image, size=(224, 224,12)):
        '''
        Resizes a black and white image to the specified size,
        adding padding to preserve the aspect ratio.
        '''
        # Get the height and width of the image
        #print(image.shape)
        if len(image.shape)>2:
            height, width, channels = image.shape
        else:
            height, width = image.shape


        # Calculate the new height and width after resizing to (224,224)
        new_height, new_width, new_channels = size


        # Calculate the number of rows/columns to add as padding
        padding_rows = (size[0] - height)
        padding_cols = (size[1] - width)

        # Add the resized image to the padded image, with padding on the left and right sides
        if len(image.shape) > 2:
            padded_image = np.zeros(size)
            padded_image[padding_rows:new_height+1, padding_cols:new_width+1,:] = image
        else:
            padded_image = np.zeros((new_height, new_width),dtype=np.uint8)
            padded_image[padding_rows: new_height+1, padding_cols:new_width+1] = image.copy()
        return padded_image

    def reduce_padding(self, image, true):
        _, height, width, _ = image.shape
        new_h, new_w=true.shape
        padding_rows = (height- new_h)
        padding_cols = (width- new_w)
        padded_image=image[:, padding_rows:height, padding_cols:width,:]
        return padded_image




    def changeValueMask(self, in_path,out_path):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr = np.load(os.path.join(in_path, file))
                arr[arr == 255] = 1
                print(np.unique(arr))
                np.save(os.path.join(out_path, file) , arr)











