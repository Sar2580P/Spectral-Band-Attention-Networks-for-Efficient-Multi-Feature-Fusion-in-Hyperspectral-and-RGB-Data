import pandas as pd
import numpy as np
import math, os
from sklearn.metrics import roc_curve ,auc, classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
#_______________________________________________________________________________________________________________________
#ROC-AUC

def plot_roc(val_label,decision_val, caption='ROC Curve'):
    num_classes=np.unique(val_label).shape[0]
    classes = []
    for i in range(num_classes):
        classes.append(i)
    plt.figure()
    decision_val = label_binarize(decision_val, classes=classes)

    if num_classes!=2:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            y_val = label_binarize(val_label, classes=classes)
            fpr[i], tpr[i], _ = roc_curve(y_val[:, i], decision_val[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i+1, roc_auc[i]))
    else:
        fpr,tpr,_ = roc_curve(val_label,decision_val, pos_label=1)
        roc_auc = auc(fpr,tpr)*100
        plt.plot(fpr,tpr,label='ROC curve (AUC=%0.2f)'%roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(caption)
    plt.legend(loc="lower right")
    plt.savefig(f"pics/ROC CURVE_classes-{num_classes}.png",dpi=300)

#_______________________________________________________________________________________________________________________
def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction

#_______________________________________________________________________________________________________________________
def metrics(labels,predictions,classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes,digits = 4))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("\nBalanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))

#_______________________________________________________________________________________________________________________
def fuzzy_rank(CF, top):
    R_L = np.zeros(CF.shape)
    for i in range(CF.shape[0]):
        for j in range(CF.shape[1]):
            for k in range(CF.shape[2]):
                R_L[i][j][k] = 1 - math.exp(-math.exp(-2.0*CF[i][j][k]))  #Gompertz Function

    K_L = 0.632*np.ones(shape = R_L.shape) #initiate all values as penalty values
    for i in range(R_L.shape[0]):
        for sample in range(R_L.shape[1]):
            for k in range(top):
                a = R_L[i][sample]
                idx = np.where(a==np.partition(a, k)[k])
                #if sample belongs to top 'k' classes, R_L =R_L, else R_L = penalty value
                K_L[i][sample][idx] = R_L[i][sample][idx]

    return K_L

#_______________________________________________________________________________________________________________________
def CFS_func(CF, K_L):
    H = CF.shape[0] #no. of classifiers
    for f in range(CF.shape[0]):
        for i in range(CF.shape[1]):
            idx = np.where(K_L[f][i] == 0.632)
            CF[f][i][idx] = 0
    CFS = 1 - np.sum(CF,axis=0)/H
    return CFS

#_______________________________________________________________________________________________________________________
def soft_max(mat):

    exps = np.exp(mat)
    sum_exps = np.sum(exps , axis = 1)
    sum_exps = np.reshape(sum_exps , (sum_exps.shape[0] , 1))
    #   print(exps.shape , sum_exps.shape)
    a =  exps / sum_exps
    #   print(np.sum(a , axis=1))
    # print(a)
    return a


#_______________________________________________________________________________________________________________________
def Gompertz(argv, num_classes, top = 2):
    L = 0 #Number of classifiers
    for arg in argv:
        L += 1

    CF = np.zeros(shape = (L,argv[0].shape[0], num_classes))

    for i, arg in enumerate(argv):
        if not isinstance(arg, np.ndarray):
            arg = arg.to_numpy()
        arg = soft_max(arg)
        CF[i][:][:] = arg

    R_L = fuzzy_rank(CF, top) #R_L is with penalties
    # print(R_L)
    RS = np.sum(R_L, axis=0)
    CFS = CFS_func(CF, R_L)
    FS = RS*CFS

    predictions = np.argmin(FS,axis=1)
    return predictions

#_______________________________________________________________________________________________________________________


def get_ensemble_performance(num_classes:int, type : str = 'train'):
    BASE_DIR = f"results/ensemble/base_models/classes-{num_classes}"
    assert os.path.exists(BASE_DIR), f"Source directory {BASE_DIR} does not exist"
    top = 30   # top 'k' classes
    classes = []
    for i in range(num_classes):
        classes.append(str(i+1))

    for fold in tqdm(range(5) ,desc = f"Gompertz Ensemble fold"):
        true_labels = None
        p = []
        data_dir = os.path.join(BASE_DIR , f"fold_{fold}")
        files = [f for f in os.listdir(data_dir) if f.endswith(f'{type}.npy')]
        files.sort()    # sorting to maintain cossistency in base models ordering
        for base_prediction_file in files:
            mat = np.load(os.path.join(data_dir, base_prediction_file))
            p.append( mat[:, :-1] )
            true_labels = mat[:, -1] if true_labels is None else true_labels

        predictions = Gompertz(top=top,num_classes=num_classes , argv = tuple(p))
        correct = np.where(predictions == true_labels)[0].shape[0]
        total = true_labels.shape[0]
        acc = correct/total

        print(f"{fold} : Accuracy =   ",acc*100)

        # metrics(true_labels,predictions,classes)



if __name__=="__main__":
    get_ensemble_performance(96)
