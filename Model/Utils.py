import numpy as np
from Preprocessing import Preprocessing
def res(cm):
    '''
    tp = cm[1][1]  # attacks true
    fn = cm[1][0]  # attacs predict normal
    fp = cm[0][1]  # normal predict attacks
    tn = cm[0][0]  # normal as normal
    '''
    if (len(cm) > 1):
        tn, fp, fn, tp=cm.ravel()

    else:
        tn=cm[0][0]
        tp=0
        fp=0
        fn=0

    attacks = tp + fn
    normals = fp + tn
    OA = (tp + tn) / (attacks + normals)
    if (len(cm) > 1):
        AA = ((tp / attacks) + (tn / normals)) / 2
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F1 = 2 * ((P * R) / (P + R))
        TPR = tp / (tp + fn)
    else:
        AA = (0 + (tn / normals)) / 2
        P=0
        R=0
        F1=0
        TPR=0
    FAR = fp / (fp + tn)

    r = [OA, AA, P, R, F1, FAR, TPR]
    return r

def predictionWithResize(path,images, model, input_shape=(32,32,12)):
    allTrue=[]
    allPred=[]
    allPredNoRounded=[]
    import os
    masks = []
    for root, _, files in os.walk(path):
        files.sort()
        for file in files:
            m = np.load(os.path.join(root, file))
            masks.append(m)
    print(len(masks))
    for i in range(len(images)):
        pred = model.predict(images[i].reshape(1, input_shape[0], input_shape[1], input_shape[2]))
        true=masks[i]
        prep=Preprocessing()
        pred = prep.reduce_padding(pred, true)
        allPredNoRounded.append(pred.ravel())
        pred = ((pred > 0.5) + 0).ravel()
        true = true.flatten()
        allTrue.append(true)
        allPred.append(pred)
    allPred = [arr.tolist() for arr in allPred]
    allPred=flattenList(allPred)
    allTrue = [arr.tolist() for arr in allTrue]
    allTrue = flattenList(allTrue)
    allPredNoRounded = [arr.tolist() for arr in allPredNoRounded]
    allPredNoRounded = flattenList(allPredNoRounded)
    return allTrue,allPred, allPredNoRounded


def predictionWithResizeNoL(path,images, model, input_shape=(32,32,12)):
    allTrue=[]
    allPred=[]
    allPredNoRounded=[]
    import os
    masks = []
    for root, _, files in os.walk(path):
        files.sort()
        for file in files:
            m = np.load(os.path.join(root, file))
            masks.append(m)
    for i in range(len(images)):
        pred = model.predict(images[i].reshape(1, input_shape[0], input_shape[1], input_shape[2]))
        true=masks[i]
        prep=Preprocessing()
        pred = prep.reduce_padding(pred, true)
        allPredNoRounded.append(pred.ravel())
        pred = ((pred > 0.5) + 0).ravel()
        true = true.flatten()
        allTrue.append(true)
        allPred.append(pred)
    return allTrue,allPred, allPredNoRounded


def flattenList(l):
    return [item for sublist in l for item in sublist]




def print_results(path, y_true, y_pred,  model, non_rounded_y_pred=None):
    """
      FUnction saving the classification report of the single outputs classifiers in  .txt format
      :param path: path to save the report
      :param y_true: true y
      :param y_pred: vpredicted y
      :param write: booleano, if true write the file, otherwise return only the classification report
      :return: classification report if write equal to false otherwise void
      """
    from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, roc_auc_score, roc_curve

    cm = confusion_matrix(y_true, y_pred)


    val = ''
    val = val + ('\n****** ******\n\n')
    val = val + (classification_report(y_true, y_pred))
    val = val + '\n\n----------- f1 macro ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='macro'))
    val = val + '\n\n----------- f1 micro ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='micro'))
    val = val + '\n\n----------- f1 weighted ---------------\n'
    val = val + str(f1_score(y_true, y_pred, average='weighted'))
    val = val + '\n\n----------- OA ---------------\n'
    val = val + str(accuracy_score(y_true, y_pred,))
    val = val + '\n\n----------- Confusion matrix ---------------\n'
    val = val + str(cm)
    val = val + '\n\n----------- tn, fp, fn, tp ---------------\n'
    val = val + str(cm.ravel())
    val = val + '\n\n----------- IOU ---------------\n'
    tn, fp, fn, tp = cm.ravel()
    val = val + str(tp / (tp + fn + fp))

    r=res(cm)
    val = val + '\n\n----------- OA  manually ---------------\n'
    val = val + str(r[0])
    val = val + '\n\n----------- AA  manually ---------------\n'
    val = val + str(r[1])
    val = val + '\n\n----------- P  attack ---------------\n'
    val = val + str(r[2])
    val = val + '\n\n----------- R  attack ---------------\n'
    val = val + str(r[3])
    val = val + '\n\n----------- F1  attack ---------------\n'
    val = val + str(r[4])
    val = val + '\n\n----------- FAR ---------------\n'
    val = val + str(r[5])
    val = val + '\n\n----------- TPR ---------------\n'
    val = val + str(r[6])



    if non_rounded_y_pred is not None:
        fpr, tpr, thresholds = roc_curve(y_true, non_rounded_y_pred)
        auc_value = roc_auc_score(y_true, non_rounded_y_pred)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(fpr, tpr, linestyle='-', marker='.', label="(auc = %0.4f)" % auc_value)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        fig.savefig(path+'auc-roc.png', bbox_inches='tight')
        val = val + '\n\n----------- AUC-ROC ---------------\n'
        val = val + str(auc_value)


    with open(path + '_results.txt', 'w', encoding='utf-16') as file:
        file.write(val)





def predictionWithResizeMulti(path, images, model, input_shape=(32, 32, 12)):
    allTrue = []
    allPred = []
    allPredNoRounded = []
    import os
    masks = []
    for root, _, files in os.walk(path):
        files.sort()
        for file in files:
            m = np.load(os.path.join(root, file))
            masks.append(m)
    print(len(masks))
    for i in range(len(masks)):
        im1= images[0][i]
        im2=images[1][i]
        im1=np.expand_dims(im1, axis=0)
        im2=np.expand_dims(im2, axis=0)
        #print(im1.shape)
        #print(im2.shape)
        #print("essc")
        #exit()
        pred = model.predict([im1,im2])
        true = masks[i]
        prep = Preprocessing()
        pred = prep.reduce_padding(pred, true)
        allPredNoRounded.append(pred.ravel())
        pred = ((pred > 0.5) + 0).ravel()
        true = true.flatten()
        allTrue.append(true)
        allPred.append(pred)
    allPred = [arr.tolist() for arr in allPred]
    allPred = flattenList(allPred)
    allTrue = [arr.tolist() for arr in allTrue]
    allTrue = flattenList(allTrue)
    allPredNoRounded = [arr.tolist() for arr in allPredNoRounded]
    allPredNoRounded = flattenList(allPredNoRounded)
    return allTrue, allPred, allPredNoRounded


def predictionWithResizeMultiNoL(path, images, model, input_shape=(32, 32, 12)):
    allTrue = []
    allPred = []
    allPredNoRounded = []
    import os
    masks = []
    for root, _, files in os.walk(path):
        files.sort()
        for file in files:
            m = np.load(os.path.join(root, file))
            masks.append(m)
    print(len(masks))
    for i in range(len(masks)):
        im1= images[0][i]
        im2=images[1][i]
        im1=np.expand_dims(im1, axis=0)
        im2=np.expand_dims(im2, axis=0)
        #print(im1.shape)
        #print(im2.shape)
        #print("essc")
        #exit()
        pred = model.predict([im1,im2])
        true = masks[i]
        prep = Preprocessing()
        pred = prep.reduce_padding(pred, true)
        allPredNoRounded.append(pred.ravel())
        pred = ((pred > 0.5) + 0).ravel()
        true = true.flatten()
        allTrue.append(true)
        allPred.append(pred)

    return allTrue, allPred, allPredNoRounded



def mergeImage(image, Y_predicted):
    y, x = image.shape
    xi = 0
    yi = 0
    imgAll = np.zeros((y, x))


    if int((y % 32)) == 0:
        rows = int((y / 32))
    else:
        rows = int((y / 32)) + 1

    if int((x % 32)) == 0:
        columns = int((x / 32))
    else:
        columns = int((x / 32)) + 1


    i = 0
    cone = 0
    y_offset = 0

    for j in range(0, rows):

        while xi < x:
            print(xi)
            print(i)
            item = Y_predicted[i]
            print(item.shape)

            cone = cone + np.count_nonzero(item == 1)

            if x - xi < 32:
                x_offset = x - xi
            else:
                x_offset = 32

            if y - yi < 32:
                y_offset = y - yi
            else:
                y_offset = 32

            Y_pred = np.resize(item, (y_offset, x_offset))

            imgAll[yi:yi + y_offset, xi:xi + x_offset] = Y_pred
            xi = xi + x_offset
            i = i + 1
        xi = 0
        yi = yi + y_offset

    print(imgAll.shape)
    return imgAll