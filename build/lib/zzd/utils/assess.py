import numpy as np
from sklearn import metrics
import warnings

def multi_scores(y_true: int, y_pred: float, threshold=0.5, show=True, show_index=True, abbr=False):
    """
    multi scores for binnary class.

    y_true:true label, list or array.
    y_prob:pred label with probility, list or array.
    threshold:Negative if y_pred < threshold, positive if y_pred > threshold. 
    show:binnary, print result or not.
   
    (1) first layer score
        TP : true positive
        TN : true negative
        FP : false positive
        FN : false engative

    (2) second layer score
        precision   = TP/(TP+FP)
        recall      = TP/(TP+FN)   
        sensitivity = TP/(TP+FN)   
        specificity = TN/(TN+FP)
        Accuracy    = (TP+TN)/(TP+TN+FP+FN)  

    (3) third layer score
         mcc = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
         f1 = 2*(precision*recall)/(precision+recall)

    (4) area score
        auc : Area under the curve of ROC(receipt operator curve) 
        auprc:Area under the precision recall curve
        ap:   Average precision-recall score

    more info:
        TPR = TP/(TP+FN) :true positive rate, Recall, sensitivity
        TNR = TN/(TN+FP) :true negative rate, specificity
        FPR = FP/(TN+FP) :false positive rate
        FNR = FN/(TP+FN) :false negative rate

        PPV = TP/(TP+FP):positive predictive value
        NPV = TN/(TN+FN):negative predictive value

        PPV=precision
        TPR=Recall=sensitivity
        TNR=specificity

    example:

    test = multi_scores(
            y_true = [0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1  ],
            y_pred = [0., 0.2, 0.4, 0.6, 0.8, 1., 0., 0.2, 0.4, 0.6, 0.8, 1],
            show=True,
            )

    """
    np.set_printoptions(suppress=True)
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always')

    y_true = np.array(y_true, float).ravel()
    y_pred = np.array(y_pred, float).ravel()

    if max(y_true) > (1.+1e-8) or min(y_true) < 0.:
        raise Exception("y_true label not in range (0, 1)!")

    if max(y_pred) > (1.+1e-8) or min(y_pred) < 0.:
        raise Exception("y_prob value not in range (0, 1)!")

    y_true_label = np.round(y_true, 0) #float to integer
    y_pred_label = np.round(y_pred, 0) #float to integer

    TP = sum((y_true > threshold) & (y_pred > threshold))
    TN = sum((y_true <= threshold) & (y_pred <= threshold))
    FP = sum((y_true <= threshold) & (y_pred > threshold))
    FN = sum((y_true > threshold) & (y_pred <= threshold))

    # PPV = np.round(metrics.precision_score(y_true_label, y_pred_label), 5) # precision
    # TPR = np.round(metrics.recall_score(   y_true_label, y_pred_label), 5) # recall

    PPV = np.round(TP/(TP+FP+1e-6),4) # precision
    TPR = np.round(TP/(TP+FN+1e-6),4) # recall
    TNR = np.round(TN/(TN+FP+1e-6),4) # specificity

    acc = np.round(metrics.accuracy_score(y_true_label, y_pred_label), 5)     # accuracy
    mcc = np.round(metrics.matthews_corrcoef( y_true_label, y_pred_label), 5) # mcc
    f1 = np.round(metrics.f1_score(y_true_label, y_pred_label), 5)            # f1-score
   
    auroc = np.round(metrics.roc_auc_score(y_true, y_pred), 5)              #auroc
    auprc = np.round(metrics.average_precision_score(y_true, y_pred), 5)    #auprc
 
    scores = (TP, TN, FP, FN, PPV, TPR, TNR, acc, mcc, f1, auroc, auprc)

    
    if show:
        if not abbr and show_index:
            print(
                "TP\tTN\tFP\tFN\tprecision\trecall\tspecificity\tAcc\tMCC\tf1\tAUROC\tAUPRC")
        elif abbr and show_index:
            print("TP\tTN\tFP\tFN\tPPV\tTPR\tTNR\tAcc\tmcc\tf1\tAUROC\tAUPRC")
            
        print("\t".join([str(_) for _ in scores[:5]]) + "\t" +
              "\t".join([str(f"{_:.3f}") for _ in scores[5:]]))
    return scores


def mean_accuray(y_true, y_pred):
    y_true = np.array(y_true, float)
    y_pred = np.array(y_pred, float)

    if max(y_true) > 1 or min(y_true) < 0:
        raise Exception("label not in range (0, 1)!")

    if max(y_pred) > 1 or min(y_pred) < 0:
        raise Exception("y_prob value not in range (0, 1)!")

    y_true_label = np.round(y_true)
    y_pred_label = np.round(y_pred)
    accuracy = np.round(metrics.accuracy_score(y_true_label, y_pred_label), 5)
    return accuracy


if __name__ == "__main__":
    test = multi_scores(
        [0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1],
        [0.001, 0.2, 0.4, 0.6, 0.8, 0.999, 0.001, 0.2, 0.4, 0.6, 0.8, 0.999], show=True,abbr=False)
    test = multi_scores(
        [0,      0,   0,   0,    0,    0,   1,    1,   1,    1,   1,   1],
        [0.01, 0.2, 0.4, 0.01, 0.2, 0.40, .001, 0.2, 0.4, 0.01, 0.2, 0.4], show=True,show_index=False)
    test = multi_scores(
        [0,      0,   0,   0,    0,    0,   1,    1,   1,    1,   1,   1],
        [0.51, 0.6, 0.6, 0.61, 0.6, 0.60, .601, 0.6, 0.6, 0.61, 0.6, 0.6], show=True,show_index=False)