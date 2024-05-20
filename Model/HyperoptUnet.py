import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
import image_generator as image_generator
import image_generator_prediction as image_generator_prediction
import Utils
import satelliteUnet

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
iteration=1


    

def trainNN(trainImage,testingImage, trainMask, testMask, name, size_test, resize, shape, attention):
    print("Load dataset")
    global pathTrainImage
    pathTrainImage=trainImage
    global pathTestImage
    pathTestImage=testingImage
    global testGlobal

    global pathTrainMask
    pathTrainMask = trainMask
    global pathTestMask
    pathTestMask = testMask

 


    global testGlobal

    global resizeGlobal
    resizeGlobal=resize
    global attentionGlobal
    attentionGlobal=attention
    global shapeImages
    shapeImages=shape

  



    global Name
    Name = name
    if attentionGlobal:
        Name=Name+'_attention'
   
    test = image_generator_prediction.ImageMaskGenerator(
        images_folder=pathTestImage,
        masks_folder=pathTestMask,
        batch_size=size_test,
        nb_classes=2, split=0, train=False, resize=resize, size=shapeImages
    )

    testGlobal=test



    trials = Trials()

    hyperparams = {"batch": hp.choice("batch", [4, 8, 16,32, 64]),
                   "augmentation": hp.choice("augmentation", ["True", "False"]),
                   "learning_rate": hp.uniform("learning_rate", 0.0001, 0.01)}

    fmin(hyperopt_fcn, hyperparams, trials=trials, algo=tpe.suggest, max_evals=25)

    return best_model


def hyperopt_fcn(params):
    global SavedParameters
    global best_model
    start_time = time.time()

    print("start train")
    global iteration


    model, val = NN(pathTrainImage, pathTrainMask, params)

    time_training = time.time() - start_time

    K.clear_session()

    SavedParameters.append(val)

    global best_val_acc
    global best_test_acc
    #global best_val_loss

    SavedParameters[-1].update(
        {"time_training": time_training,  "augmentation": params["augmentation"],
         "learning_rate": params["learning_rate"], "batch": params["batch"]})


    if SavedParameters[-1]["F1_val"] > best_val_acc:
        print("new saved model:" + str(SavedParameters[-1]))
        best_model = model
        model.save(Name+'_model.h5')
        model.save(Name+'_model.tf')
        model.save_weights(Name+"_weights.h5")

        best_val_acc = SavedParameters[-1]["F1_val"]


    SavedParameters = sorted(SavedParameters, key=lambda i: float('-inf') if math.isnan(i['F1_val']) else i['F1_val'],
                             reverse=True)

    try:
        with open(Name + '_Results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")
    iteration=iteration+1
    return {'loss': -val["F1_val"], 'status': STATUS_OK}


def NN(pathTrainImage,pathTrainMask, params):
    print(params)

    model=satelliteUnet.satellite_unet(shapeImages,attention=attentionGlobal)
    model.summary()

    batch_size = params['batch']

    generator = image_generator.ImageMaskGenerator(
        images_folder=pathTrainImage,
        masks_folder=pathTrainMask,
        batch_size=batch_size,
        nb_classes=2, augmentation=params['augmentation'], resize=resizeGlobal, size=shapeImages
    )

    valid = image_generator.ImageMaskGenerator(
        images_folder=pathTrainImage,
        masks_folder=pathTrainMask,
        batch_size=batch_size,
        nb_classes=2,
        validation=True, resize=resizeGlobal,size = shapeImages
    )

    save_model_band_attention = [EarlyStopping(patience=20, verbose=1, monitor="val_loss"),
                                 ModelCheckpoint(Name + '.hdf5', monitor='val_loss', verbose=1,
                                                 save_best_only=True)]

    loss = tversky_loss
    metrics = [dice_coef, 'mse', 'accuracy']

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss=loss,
                  metrics=metrics)
    hist=model.fit(generator, epochs=150, validation_data=valid,
              callbacks=[save_model_band_attention])

    np.save(Name + '-history.npy', model.history.history)


    x_val=valid[0][0]
    y_val=(valid[0][1]).ravel()
    print(x_val.shape)
    print(y_val.shape)

    Y_predicted = model.predict(x_val, verbose=0, use_multiprocessing=True, workers=12)
    print(Y_predicted.shape)
    Y_predicted = ((Y_predicted > 0.5) + 0).ravel()
    print(Y_predicted.shape)
    print(y_val.shape)
    cfVal = confusion_matrix(y_val, Y_predicted)
    rVal = Utils.res(cfVal)

    x_train=generator[0][0]
    y_train=generator[0][1].ravel()
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
    precision_weighted_train, recall_weighted_train, fscore_weighted_train, support = precision_recall_fscore_support(y_val,
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

    return model, {"val_loss": min_val_loss, "F1_val":rVal[4],"P_val": rVal[2],"R_val": rVal[3],"TP_train": tp,
          "FN_train": fn, "FP_train": fp, "TN_train": tn, "OA_train": r[0],
         "P_train": r[2],"R_train": r[3],"F1_train": r[4],"precision_macro_train": precision_macro_train,
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
