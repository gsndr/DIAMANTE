import numpy as np
from keras import Model, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, Flatten, Lambda, Dropout, Conv2DTranspose
from keras.optimizers import Adam

import os
import csv
import math

from keras import models
from keras import layers
import keras.metrics

my_seed = 12
np.random.seed(my_seed)
import random

random.seed(my_seed)

import tensorflow as tf

tf.random.set_seed(12)
from losses import *
import numpy as np
from hyperopt import STATUS_OK
from hyperopt import tpe, hp, Trials, fmin
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import balanced_accuracy_score

from sklearn.model_selection import train_test_split

import time

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import multi_image_generator as image_generator
import multi_image_generator_prediction as image_generator_prediction
import Utils


XGlobal = []
YGlobal = []

XTestGlobal = []
YTestGlobal = []

SavedParameters = []
Mode = ""
Name = ""
best_val_acc = 0
best_val_loss = np.inf
paramsGlobal = 0
best_model = None
teacherModel = 0
trainable = 0


def trainNN(trainImage,testingImage, trainMask, testMask, name, size_test, resize, shape, shape1, attention, ch, ch1, mode, typeconcat):
    print("Load dataset")
    print(trainMask)

    global pathTrainImage
    pathTrainImage = trainImage
    global pathTestImage
    pathTestImage = testingImage
    global testGlobal




    global pathTrainMask
    pathTrainMask = trainMask
    global pathTestMask
    pathTestMask = testMask

    global testGlobal

    global resizeGlobal
    resizeGlobal = resize
    global attentionGlobal
    attentionGlobal = attention
    global shapeImages
    shapeImages = shape

    global shapeImages1
    shapeImages1 = shape1

    global channel
    channel=ch

    global channel1
    channel1=ch1

    global sum
    sum=typeconcat

    global Name
    Name = name



    global lateFusion
    lateFusion=mode
    if attentionGlobal:
        Name = Name + '_attention'
    if lateFusion==0:
        Name = Name + '_sum_'+str(sum)

    test = image_generator_prediction.ImageMaskGenerator(
        images_folder=pathTestImage,
        masks_folder=pathTestMask,
        batch_size=size_test,
        nb_classes=2, split=0, train=False, resize=resize, size=shapeImages, ch=channel
    )

    testGlobal = test

    global testGlobal1



    testGlobal1 = test


    trials = Trials()

    hyperparams = {"batch": hp.choice("batch", [4, 8, 16, 32, 64]),
                   "augmentation": hp.choice("augmentation", ["True", "False"]),
                   "learning_rate": hp.uniform("learning_rate", 0.0001, 0.01)}

    fmin(hyperopt_fcn, hyperparams, trials=trials, algo=tpe.suggest, max_evals=25)

    print("done")
    return best_model


def hyperopt_fcn(params):
    global SavedParameters
    global best_model
    start_time = time.time()

    print("start train")

    model, val = NN(pathTrainImage, pathTrainMask, params)

    time_training = time.time() - start_time

    print("start predict")

    start_time = time.time()

    if resizeGlobal:
        shapeList = list(shapeImages)
        shapeList[2] = int(channel - channel1)
        shapeImagesNew = tuple(shapeList)

        YTestGlobal, Y_predicted, _ = Utils.predictionWithResizeMulti(pathTestMask, testGlobal[0][0], model,
                                                                 input_shape=shapeImagesNew)

    else:
        Y_predicted = model.predict(testGlobal[0][0], verbose=0, use_multiprocessing=True, workers=12)
        YTestGlobal = testGlobal[0][1].ravel()
        Y_predicted = ((Y_predicted > 0.5) + 0).ravel()

    time_predict = time.time() - start_time

    precision_macro_t, recall_macro_t, fscore_macro_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                 Y_predicted,
                                                                                                 average='macro')
    precision_micro_t, recall_micro_t, fscore_micro_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                 Y_predicted,
                                                                                                 average='micro')
    precision_weighted_t, recall_weighted_t, fscore_weighted_t, support = precision_recall_fscore_support(YTestGlobal,
                                                                                                          Y_predicted,
                                                                                                          average='weighted')

    accuracy_t = accuracy_score(YTestGlobal, Y_predicted)
    cf = confusion_matrix(YTestGlobal, Y_predicted)
    r = Utils.res(cf)
    if (len(cf) > 1):
        tn, fp, fn, tp = cf.ravel()
        iou = tp / (tp + fn + fp)
    else:
        tn = cf[0][0]
        tp = 0
        fp = 0
        fn = 0
        iou=0



    K.clear_session()

    SavedParameters.append(val)

    global best_val_acc
    global best_test_acc
    # global best_val_loss

    SavedParameters[-1].update(
        {"precision_macro_t": precision_macro_t, "recall_macro_t": recall_macro_t, "fscore_macro_t": fscore_macro_t,
         "precision_micro_t": precision_micro_t, "recall_micro_t": recall_micro_t, "fscore_micro_t": fscore_micro_t,
         "precision_weighted_t": precision_weighted_t, "recall_weighted_t": recall_weighted_t,
         "fscore_weighted_t": fscore_weighted_t,
         "accuracy_t": accuracy_t, "IOU_test": iou, "TP_test": tp,
         "FN_test": fn, "FP_test": fp, "TN_test": tn,
         "time_training": time_training, "time_predict": time_predict, "augmentation": params["augmentation"],
         "learning_rate": params["learning_rate"], "batch": params["batch"]})

    SavedParameters[-1].update({
        "OA_test": r[0],
        "P_test": r[2],
        "R_test": r[3],
        "F1_test": r[4],
        "FAR_test": r[5],
        "TPR_test": r[6]})

    # if SavedParameters[-1]["val_loss"] < best_val_loss:
    if SavedParameters[-1]["F1_val"] > best_val_acc:
        print("new saved model:" + str(SavedParameters[-1]))
        best_model = model
        import os
        # model.save(Name.replace(".csv", "_model.h5"))
        model.save(Name + '_model.h5')
        model.save(Name + '_model.tf')
        model.save_weights(Name + "_weights.h5")

        '''
        for i in range(len(model.weights)):
            model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)

        model.save_weights(Name.replace(".csv", "_Dweights.h5")) # OK, saved.
        '''

        # model.save_weights(Name.replace(".csv", "_Dweights.h5"))
        best_val_acc = SavedParameters[-1]["F1_val"]

    # SavedParameters = sorted(SavedParameters, key=lambda i: i['F1_val'], reverse=True)
    SavedParameters = sorted(SavedParameters, key=lambda i: float('-inf') if math.isnan(i['F1_val']) else i['F1_val'],
                             reverse=True)

    try:
        with open(Name + '_Results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")

    # return {'loss': -val["fscore_weighted_val"], 'status': STATUS_OK}
    return {'loss': -val["F1_val"], 'status': STATUS_OK}


def NN(pathTrainImage, pathTrainMask, params):
    print(params)
    shapeList = list(shapeImages)
    shapeList[2] = int(channel - channel1)
    shapeImagesNew=tuple(shapeList)


    if lateFusion:
        print("Late fusion")
        from satelliteLateUnet import satellite_unet
        model = satellite_unet(shapeImagesNew, shapeImages1)
    else:
        print("Middle fusion")
        from satelliteFuseUnet import satellite_unet
        model = satellite_unet(shapeImagesNew, shapeImages1 sum=sum)

    model.summary()


    import CustomUnet

    # model = CustomUnet.custom_unet((32, 32, 12))
    #model=keras_segmentation.models.unet.unet(2, input_height=32, input_width=32, encoder_level=3, channels=10)

    ####implement load ###
    batch_size = params['batch']
    tf.keras.utils.plot_model(model, to_file='UnetLate.png', show_shapes=True, show_layer_names=True,expand_nested=True, show_layer_activations=True)
    exit()



    generator = image_generator.ImageMaskGenerator(
        # images_folder='../../Dataset/32_32/Train/images/',
        images_folder=pathTrainImage,
        masks_folder=pathTrainMask,
        batch_size=batch_size,
        nb_classes=2, augmentation=params['augmentation'], resize=resizeGlobal, size=shapeImages, ch=channel, ch1=channel1
    )

    valid = image_generator.ImageMaskGenerator(
        images_folder=pathTrainImage,
        masks_folder=pathTrainMask,
        batch_size=batch_size,
        nb_classes=2,
        validation=True, resize=resizeGlobal, size=shapeImages, ch=channel, ch1=channel1
    )



    save_model_band_attention = [EarlyStopping(patience=20, verbose=1, monitor="val_loss"),
                                 ModelCheckpoint(Name + '.hdf5', monitor='val_loss', verbose=1,
                                                 save_best_only=True)]

    loss = tversky_loss
    metrics = [dice_coef, 'mse', 'accuracy']

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss=loss,
                  metrics=metrics)
    hist = model.fit(generator, epochs=150, validation_data=valid,
                     callbacks=[save_model_band_attention])

    np.save(Name + '-history.npy', model.history.history)

    x_val = valid[0][0]
    y_val = (valid[0][1]).ravel()
    #print(x_val.shape)
    print(y_val.shape)

    Y_predicted = model.predict(x_val, verbose=0, use_multiprocessing=True, workers=12)
    print(Y_predicted.shape)
    Y_predicted = ((Y_predicted > 0.5) + 0).ravel()
    print(Y_predicted.shape)
    print(y_val.shape)
    cfVal = confusion_matrix(y_val, Y_predicted)
    rVal = Utils.res(cfVal)

    x_train = generator[0][0]
    y_train = generator[0][1].ravel()
    Y_predicted_train = model.predict(x_train, verbose=0, use_multiprocessing=True, workers=12)
    Y_predicted_train = ((Y_predicted_train > 0.5) + 0).ravel()
    cf = confusion_matrix(y_train, Y_predicted_train)
    r = Utils.res(cf)
    if (len(cf) > 1):
        tn, fp, fn, tp = cf.ravel()
    else:
        tn = cf[0][0]
        tp = 0
        fp = 0
        fn = 0

    precision_macro_train, recall_macro_train, fscore_macro_train, support = precision_recall_fscore_support(y_train,
                                                                                                             Y_predicted_train,
                                                                                                             average='macro')
    precision_micro_train, recall_micro_train, fscore_micro_train, support = precision_recall_fscore_support(y_val,
                                                                                                             Y_predicted,
                                                                                                             average='micro')
    precision_weighted_train, recall_weighted_train, fscore_weighted_train, support = precision_recall_fscore_support(
        y_val,
        Y_predicted,
        average='weighted')
    accuracy_train = accuracy_score(y_train, Y_predicted_train)

    precision_macro_val, recall_macro_val, fscore_macro_val, support = precision_recall_fscore_support(y_val,
                                                                                                       Y_predicted,
                                                                                                       average='macro')
    precision_micro_val, recall_micro_val, fscore_micro_val, support = precision_recall_fscore_support(y_val,
                                                                                                       Y_predicted,
                                                                                                       average='micro')
    precision_weighted_val, recall_weighted_val, fscore_weighted_val, support = precision_recall_fscore_support(y_val,
                                                                                                                Y_predicted,
                                                                                                                average='weighted')
    accuracy_val = accuracy_score(y_val, Y_predicted)

    del support
    epoches = len(hist.history['val_loss'])
    min_val_loss = np.amin(hist.history['val_loss'])

    return model, {"val_loss": min_val_loss, "F1_val": rVal[4], "P_val": rVal[2], "R_val": rVal[3], "TP_train": tp,
                   "FN_train": fn, "FP_train": fp, "TN_train": tn, "OA_train": r[0],
                   "P_train": r[2], "R_train": r[3], "F1_train": r[4], "precision_macro_train": precision_macro_train,
                   "recall_macro_train": recall_macro_train, "fscore_macro_train": fscore_macro_train,
                   "precision_micro_train": precision_micro_train, "recall_micro_train": recall_micro_train,
                   "fscore_micro_train": fscore_micro_train,
                   "precision_weighted_train": precision_weighted_train, "recall_weighted_train": recall_weighted_train,
                   "fscore_weighted_train": fscore_weighted_train,
                   "accuracy_train": accuracy_train, "precision_macro_val": precision_macro_val,
                   "recall_macro_val": recall_macro_val, "fscore_macro_val": fscore_macro_val,
                   "precision_micro_val": precision_micro_val, "recall_micro_val": recall_micro_val,
                   "fscore_micro_val": fscore_micro_val,
                   "precision_weighted_val": precision_weighted_val, "recall_weighted_val": recall_weighted_val,
                   "fscore_weighted_val": fscore_weighted_val,
                   "accuracy_val": accuracy_val, "epochs": epoches}
