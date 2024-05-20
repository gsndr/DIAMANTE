import tensorflow.keras.backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)





def accuracy_teacher(y_true, y_pred):
    y_pred = y_pred[:, :, :, 0]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = tf.cast((y_pred_f > 0.5), tf.float32)
    true_pos = K.sum(y_true_f * y_pred_f)
    true_neg= K.sum((1-y_true_f) * (1-y_pred_f))
    #tf.print("true pos :", true_pos)
    #tf.print("true neg:", true_neg)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)

def accuracy(y_true, y_pred):
    y_pred_list =0
    for j in range(y_pred.shape[3]):
        y_pred_list +=K.flatten(y_pred[:, :, :, j])

    y_pred_f=y_pred_list/y_pred.shape[3]

    y_pred_f = tf.cast((y_pred_f > 0.5), tf.float32)
    y_true_f = K.flatten(y_true)
    #y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    true_neg= K.sum((1-y_true_f) * (1-y_pred_f))
    #tf.print("true pos :", true_pos)
    #tf.print("true neg:", true_neg)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    return (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)



def f1(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_list = 0
    for j in range(y_pred.shape[3]):
        y_pred_list +=K.flatten(y_pred[:, :, :, j])
    y_pred_f=y_pred_list/y_pred.shape[3]

    y_pred_f=tf.cast((y_pred_f > 0.5),tf.float32)


    p = prec(y_true_f, y_pred_f)
    r = rec(y_true_f, y_pred_f)
    F1=2 * ((p * r) / (p + r + K.epsilon()))

    return F1

def prec(y_true, y_pred):
    true_pos = K.sum(y_true * y_pred)
    false_pos = K.sum((1 - y_true) * y_pred)
    return true_pos / ((true_pos+ false_pos) + K.epsilon())
    
def rec(y_true, y_pred):
    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(y_true * (1 - y_pred))
    return true_pos / ((true_pos+ false_neg) + K.epsilon())

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)

    # tf.print("true pos :", true_pos)
    # tf.print("false neg : ", false_neg)
    # tf.print("false pos : ", false_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)




def feature_loss_function(fea, target_fea):
    #intermediate=tf.cast(((fea > 0) | (target_fea > 0)), tf.float32)
    #tf.print(intermediate)
    return tf.norm(fea-target_fea, ord='euclidean')
    #loss = (tf.math.abs(fea - target_fea)**2)
    #return tf.math.sqrt(loss)

