import os
import numpy as np
from skimage import io
import tifffile as tiff
from tiler import Tiler, Merger


class PreprocessingImages():
    def __init__(self, tilesSizeImage, tileSizeMaks, resizeChannel, listToremove, tiles=1):
        self.tiles=tiles
        self.tileSize=tilesSizeImage
        self.tileMask=tileSizeMaks
        self.resizeChannel=resizeChannel
        self.toremove=listToremove
        self.scaler = None




    def images(self, pathImages, train):
        if os.path.exists(pathImages):
            print('The file exists!')
        else:
            print('The file does not exist.')

       
        newPath=pathImages+'Numpy/'
        self.tiffToNumpy(pathImages+'Tiff/', newPath)
        self.check_nan(newPath, newPath)
        if self.resizeChannel:
            self.removeBand(newPath,newPath, self.toremove)

        print("Scale Min Max")
        if train:
            self.min_max(pathImages + 'Numpy/', pathImages + 'Numpy/', train=1)
        else:
            self.min_max(pathImages + 'Numpy/', pathImages + 'Numpy/', train=0)

        if self.tiles:
            self.get_tiles(newPath,pathImages+'Tiles/', self.tileSize)





                
                
    def masks(self, path, changeDict, ds):
        newPath = path + 'Numpy/'
        self.tiffToNumpy(path+'Tiff/',newPath)
        self.check_nan(newPath, newPath)
        self.changeValueMask(newPath, newPath, changeDict)
        if self.tiles:
            self.get_tiles(newPath, path+'Tiles/', self.tileMask)


    def get_tiles(self,in_path, outpath, tile_shape):
        self.deleteTiles(outpath)
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                image = np.load(os.path.join(in_path, file))
                # Setup tiling parameters
                print(image.shape)
                tiler = Tiler(data_shape=image.shape,
                              tile_shape=tile_shape, mode='irregular',
                              channel_dimension=None)
                for tile_id, tile in tiler(image):
                    print(f'Tile {tile_id} out of {len(tiler)} tiles.')
                    savefile = os.path.splitext(file)[0] + '_' + str(tile_id)
                    np.save(os.path.join(outpath, savefile), tile)



    def tiffToNumpy(self, in_path, out_path):
        print(in_path)
        if os.path.exists(in_path):
            print('The file exists!')
        else:
            print('The file does not exist.')
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                imgarr = io.imread(os.path.join(root, file))
                np.save(os.path.join(out_path, file)+'.npy', imgarr)


    def check_nan(self, in_path, out_path):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr = np.load(os.path.join(in_path, file))
                np.nan_to_num(arr, copy=False)
                np.save(os.path.join(out_path, file), arr)

    def NumpytoTiff(self, in_path, out_path):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr=np.load(os.path.join(in_path, file))
                arr = np.asarray(arr)
                arr=arr*255
                print(type(file))
                tiff.imwrite( os.path.join(out_path, file)+'.tiff', arr)

    def removeBand(self, in_path, out_path, bands: list):
        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr=np.load(os.path.join(in_path, file))

                arr = np.delete(arr, bands, axis=2)
                np.save(os.path.join(out_path, file), arr)






    def changeValueMask(self, in_path,out_path, dictChange):

        for root, _, files in os.walk(in_path):
            files.sort()
            for file in files:
                arr = np.load(os.path.join(in_path, file))
                arr[arr == dictChange[1]] = 1
                arr[arr == np.nan] = 0
                print(np.unique(arr))
                np.save(os.path.join(out_path, file) , arr)




        
    def deleteTiles(self, directory):
        for root, _, files in os.walk(directory):
            files.sort()
            for f in files:
                os.remove(os.path.join(directory, f))




    def min_max(self,in_path,out_path, train=0):
        from sklearn.preprocessing import MinMaxScaler
        import os
        import numpy as np
        list_of_images=[]

        if train:
            for root, _, files in os.walk(in_path):
                files.sort()

            for file in files:
                arr = np.load(os.path.join(in_path, file))
                # Convert image to NumPy array
                img_array = np.array(arr)

                # Flatten the image array
                flat_img_array = img_array.flatten()
                list_of_images.append(flat_img_array )



            # Convert the list of images to a 2D array
                # Concatenate all image data into a single array
            images_array = np.concatenate(list_of_images).reshape(-1, 1)
            print(images_array.shape)



            # Initialize the MinMaxScaler
            scaler = MinMaxScaler()

            # Fit the scaler to the data and transform the data
            self.scaler = scaler.fit(images_array)


        for root, _, files in os.walk(in_path):
            files.sort()

        for file in files:
            arr = np.load(os.path.join(in_path, file))
            print(arr.shape)
            flat_img_array = arr.flatten().reshape(-1, 1)

            # Apply Min-Max scaling
            scaled_img_array = self.scaler.transform(flat_img_array).reshape(arr.shape)
            print(scaled_img_array.shape)

            np.save(os.path.join(out_path, file), scaled_img_array)










