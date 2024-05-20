import tensorflow as tf
import os
import datetime

my_seed = 42

import random
import sys
import configparser

random.seed(my_seed)

tf.random.set_seed(my_seed)
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ['PYTHONHASHSEED'] = '1'
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import Utils

import HyperoptUnet
from tensorflow.keras.models import load_model


from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tensorflow.python.client import device_lib

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


def main():
    dataset = sys.argv[1]
    config = configparser.ConfigParser()
    config.read('CONFIG1.conf')
    # this contains path dataset and models
    dsConf = config[dataset]
    # this contain the variable related to the flow
    settings = config['setting']



    pathTrainImage = dsConf.get('pathDatasetTrain')
    pathTestImage = dsConf.get('pathDatasetTest')

    pathTrainMask = dsConf.get('pathDatasetTrainM')
    pathTestMask = dsConf.get('pathDatasetTestM')

    pathModel = dsConf.get('pathModels')
    shape = int(dsConf.get('shape'))
    ch = int(dsConf.get('channels'))
    ch1 = int(dsConf.get('channels1'))
    tiles = int(dsConf.get('tiles'))
    tilesSize = int(dsConf.get('tilesSize'))

    resize = int(settings.get('RESIZE'))


    attack = int(dsConf.get('attack'))
    scale = int(settings.get('SCALE'))

    late = int(settings.get('LATE'))
    sum = int(settings.get('SUM'))

    dictChange = {1: attack}
    ds = dsConf.get('ds')

    # pathModel=pathModel+ 'exp_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # os.makedirs(pathModel,mode=0o777, exist_ok=True )
    # os.chmod(pathModel, 0o777)

    #Define the name of the model according with the configuration selected
    name_model = pathModel + 'unet_resize_' + str(resize)+'_scale'+str(scale)
    if (int(settings.get('TRAIN_LATEUNET')) == 1):
        if late:
            name_model = pathModel + 'unet_resize_' + str(resize) + '_scale' + str(scale)+'_lateUnet'
        else:
            name_model = pathModel + 'unet_resize_' + str(resize) + '_scale' + str(scale) + '_hybridUnet'
    rChannel = int(dsConf.get('resizeChannel'))




    if (int(settings.get('PREPROCESSING')) == 1):
        from PreprocessingImages import PreprocessingImages
        preprocessing=PreprocessingImages((tilesSize,tilesSize,ch), (tilesSize,tilesSize), rChannel, [12], tiles=1, scale=scale)

        preprocessing.images(pathTrainImage, True, pathTestImage)
        preprocessing.images(pathTestImage, False)
        if (int(settings.get('PREPROCESSING_MASKS')) == 1):
            preprocessing.masks(pathTrainMask, dictChange, ds)
            preprocessing.masks(pathTestMask, dictChange, ds)

    if tiles:
        pathTrainImage = pathTrainImage + 'Tiles/'
        pathTestImage = pathTestImage + 'Tiles/'
        pathTrainMask = pathTrainMask + 'Tiles/'
        pathTestMask = pathTestMask + 'Tiles/'
    else:
        pathTrainImage = pathTrainImage + 'Numpy/'
        pathTestImage = pathTestImage + 'Numpy/'
        pathTrainMask = pathTrainMask + 'Numpy/'
        pathTestMask = pathTestMask + 'Numpy/'


    lst = os.listdir(pathTestMask)  # your directory path
    size_test = len(lst)
    print(size_test)

    if (int(settings.get('LOAD_NN')) == 0):
        print('Train teacher')
        # Train U-Net with generator
        teacher = HyperoptUnet.trainNN(pathTrainImage, pathTestImage, pathTrainMask, pathTestMask, name_model,
                                       size_test, resize, (shape, shape, ch))

    elif (int(settings.get('LOAD_NN')) == 1):
        print('Load teacher')

        teacher = load_model(dsConf.get('pathModels') + dsConf.get('nameModel'), compile=False)
        print("teacher summary'")
        teacher.summary()



    if (int(settings.get('TRAIN_LATEUNET')) == 1):
        print('Train late fusion')
        # Train U-Net with generator
        import HyperoptLateUnet
        teacher = HyperoptLateUnet.trainNN(pathTrainImage, pathTestImage, pathTrainMask, pathTestMask, name_model,
                                       size_test, resize, (shape, shape, ch),(shape, shape, ch1), attention=attention, ch=ch, ch1=ch1, mode=late, typeconcat=sum)





    if (int(settings.get('PREDICTION')) == 1):
        import numpy as np

        name_model = pathModel + 'unet_resize_' + str(resize) + '_scale' + str(scale)
        if (int(settings.get('PREDICT_LATEUNET')) == 1):
            if late:
                name_model = pathModel + 'unet_resize_' + str(resize) + '_scale' + str(scale) + '_lateUnet'
            else:
                name_model = pathModel + 'unet_resize_' + str(resize) + '_scale' + str(scale) + '_hybridUnet'
                name_model = name_model + '_sum_' + str(sum)



        print(name_model)
        #name_model = 'Models/Sentinel_2_1//unet_resize_1_scale2_model.tf'
        # name_model = 'Models/Baptistev2/unet_resize_1_model.tf'
        # name_model = 'Models/Baptistev2/unet_resize_1_model.tf'
        resizeChannel = False
        pathImagesSave = 'Images/'  # qui

        from losses import dice_coef_self, accuracy_teacher, f1, accuracy
        model = tf.keras.models.load_model(name_model+'_model.tf', compile=False,
                                           custom_objects={"dice_coef_self": dice_coef_self, "accuracy_teacher": f1,
                                                           "f1": accuracy_teacher, "accuracy": accuracy})

        #from keras.utils.vis_utils import plot_model
        model.summary()
        #plot_model(model, to_file='self.png', show_shapes=True, show_layer_names=True)

        import image_generator_prediction
        import Utils

        col = ['tn', 'fp', 'fn', 'tp']
        import pandas as pd
        df = pd.DataFrame(columns=col)
        


        #pathTestMask = '../SWIFTT/DS/Masks/Test1/Tiles/'
        #pathTestImage = '../SWIFTT/DS/sentinel_1_2/Test1/Tiles/'

        if (int(settings.get('PREDICT_LATEUNET')) == 1):
            import multi_image_generator_prediction as image_generator_prediction
            test = image_generator_prediction.ImageMaskGenerator(
                images_folder=pathTestImage,
                masks_folder=pathTestMask,
                batch_size=size_test,
                nb_classes=2, split=0, train=False, resize=resize, size=(shape, shape, ch), ch=ch
            )

        else:
            import image_generator_prediction as image_generator_prediction
            test = image_generator_prediction.ImageMaskGenerator(
                images_folder=pathTestImage,
                masks_folder=pathTestMask,
                batch_size=size_test,
                nb_classes=2, split=0, train=False, resize=resize, size=(shape, shape, ch))



        print(pathTestMask)
        lst = os.listdir(pathTestMask)  # your directory path
        size_test = len(lst)
        print(size_test)



        if resize:
            if (int(settings.get('PREDICT_LATEUNET')) == 1):
                newchannel = int(ch - ch1)

                YTestGlobal, Y_predicted, _ = Utils.predictionWithResizeMulti(pathTestMask, test[0][0], model,
                                                                              input_shape=(shape, shape,newchannel))
            else:
                YTestGlobal, Y_predicted, _ = Utils.predictionWithResize(pathTestMask, test[0][0], model,input_shape=(shape, shape, ch))

        else:
            Y_predicted = model.predict(test[0][0], verbose=0, use_multiprocessing=True, workers=12)
            YTestGlobal = test[0][1].ravel()
            Y_predicted = ((Y_predicted > 0.5) + 0).ravel()

        Y_predicted = Y_predicted  # here
        Y_prob = Y_predicted
       

        Utils.print_results(name_model, YTestGlobal, Y_predicted, model, Y_prob)
        cm = confusion_matrix(YTestGlobal, Y_predicted)
        print(Utils.res(cm))



        thisdict = {
            '78': 81,
            '79': 28,
            '80': 36,
            '81': 70,
            '82': 54,
            '83': 4,
            '84': 36,
            '85': 42,
            '86': 12,
            '87': 24,
            '88': 6,
            '89': 42,
            '90': 24,
            '91': 72,
            '92': 6,
            '93': 6,
        }

        #vito
        thisdict = {
            '78': 36,
            '79': 15,
            '80': 16,
            '81': 35,
            '82': 24,
            '83': 1,
            '84': 16,
            '85': 25,
            '86': 6,
            '87': 12,
            '88': 4,
            '89': 20,
            '90': 12,
            '91': 36,
            '92': 4,
            '93': 4,
        }
        

        thisdict = {
            '30': 9,
            '31': 35,
            '32': 96,
            '33': 9,
            '34': 12,

        }
        # vitogee
        thisdict = {
            '78': 81,
            '79': 28,
            '80': 36,
            '81': 70,
            '82': 54,
            '83': 4,
            '84': 36,
            '85': 49,
            '86': 12,
            '87': 24,
            '88': 6,
            '89': 42,
            '90': 24,
            '91': 72,
            '92': 6,
            '93': 6,

        }
        
        


        for k, v in thisdict.items():
            print(k)
            pred = []
            true = []
            ##Ricorda di cambiare 2 con 02, 3 con 03 nella cartella dei tiles

            pathSingle= dsConf.get('pathSingleMask')

            pathMask = pathSingle + k + '/'
            pathSingle= dsConf.get('pathSingleImages')
            pathImages = pathSingle + k + '/'  # qui
            # pathImages = 'Data/Baptiste/v2/' + 'Tiling1/' + k + '/'
            # pathMask = 'Data/Baptiste/v2/' + 'masks/' + k + '/'  # qui
            # pathMask = 'Data/' + 'Fires/MaskSingle/' + k + '/'
            # pathImages = 'Data/' + 'Fires/ImagesSingle/' + k + '/'  # qui
            print(pathMask)
            print(pathImages)
            if (int(settings.get('PREDICT_LATEUNET')) == 1):
                import multi_image_generator_prediction as image_generator_prediction
                test = image_generator_prediction.ImageMaskGenerator(
                    images_folder=pathImages,
                    masks_folder=pathMask,
                    batch_size=v,
                    nb_classes=2, split=0, train=False, resize=resize, size=(shape, shape, ch), ch=ch
                )

            else:
                import image_generator_prediction as image_generator_prediction
                test = image_generator_prediction.ImageMaskGenerator(
                    images_folder=pathImages,
                    masks_folder=pathMask,
                    batch_size=v,
                    nb_classes=2, split=0, train=False, resize=resize, size=(shape, shape, ch))

            import Utils
            import numpy as np
            import imageio
            pathOriginalMask=dsConf.get('pathDatasetTestM')
            image = np.load(pathOriginalMask+'Numpy/mask_' + k + '.tif.npy')
            # image = np.load('Data/Fires/masks/Numpy/' + k + '_mask.tiff.npy')
            # image = np.load('Data/Baptiste/v2/masktif/mask_' + k + '.tif.npy')
            print(image.shape)

            # x,y=image.shape
            y, x = image.shape  # bark beetle
            print(image.shape)

            if resize:
                if (int(settings.get('PREDICT_LATEUNET')) == 1):
                    newchannel = int(ch - ch1)

                    YTestGlobal, Y_predicted, _ = Utils.predictionWithResizeMultiNoL(pathMask, test[0][0], model,
                                                                                  input_shape=(
                                                                                  shape, shape, newchannel))
                else:
                    YTestGlobal, Y_predicted, _ = Utils.predictionWithResizeNoL(pathMask, test[0][0], model,
                                                                             input_shape=(shape, shape, ch))

            else:
                Y_predicted = model.predict(test[0][0], verbose=0, use_multiprocessing=True, workers=12)
                YTestGlobal = test[0][1].ravel()
                Y_predicted = ((Y_predicted > 0.5) + 0).ravel()



            print(len(Y_predicted))
            print(len(YTestGlobal))


            pred = [arr.tolist() for arr in Y_predicted]  #perch?
            pred = Utils.flattenList(pred)

            yt = [arr.tolist() for arr in YTestGlobal]
            yt = Utils.flattenList(yt)

            c = confusion_matrix(yt, pred)
            if (len(c) > 1):
                tn, fp, fn, tp = c.ravel()
                r = [tn, fp, fn, tp]
            else:
                r = [c[0][0], 0, 0, 0]

            res = pd.DataFrame([r], columns=col)
            df = pd.concat([df, res], ignore_index=True)
            #df = df.append(res, ignore_index=True)
            # Y_predicted=np.array(Y_predicted)

            i = 0
            xi = 0
            yi = 0
            imgAll = np.zeros((y, x))
            print(imgAll.shape)

            if int((y % 32)) == 0:
                rows = int((y / 32))
            else:
                rows = int((y / 32)) + 1

            if int((x % 32)) == 0:
                columns = int((x / 32))
            else:
                columns = int((x / 32)) + 1
            loop = columns * rows

            i = 0
            cone = 0
            x_offset = 0
            y_offset = 0

            for j in range(0, rows):

                while xi < x:
                    print(xi)
                    print(i)
                    item = Y_predicted[i]
                    print(item.shape)

                    cone = cone + np.count_nonzero(item == 1)
                    one = np.count_nonzero(item == 1)

                    print('tot 1', cone)
                    if x - xi < 32:
                        x_offset = x - xi
                    else:
                        x_offset = 32

                    if y - yi < 32:
                        y_offset = y - yi
                    else:
                        y_offset = 32

                    Y_pred = np.resize(item, (y_offset, x_offset))
                    # if one > 0: #togleire if?
                    # Y_pred=item
                    # print('hre')
                    print('1 in item from x:' + str(xi) + ' to y' + str(yi) + ' _' + str(j) + 'no ' + str(
                        one))

                    imgAll[yi:yi + y_offset, xi:xi + x_offset] = Y_pred
                    print('in all images', str(np.count_nonzero(imgAll == 1)))
                    xi = xi + x_offset
                    i = i + 1
                xi = 0
                yi = yi + y_offset
                print('x1')
                print(yi)

            print(imgAll.shape)
            print(pathImagesSave)

            imageio.imwrite(pathImagesSave + k + '.png', (imgAll * 255).astype(np.uint8))

        df.to_csv(name_model+'_Singleresults.csv', index=False)


if __name__ == "__main__":
    main()
