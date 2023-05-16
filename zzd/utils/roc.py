import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from sklearn import metrics


def roc_curve(data_list, labels=None, title=None, colors=None, save_file=False):
    """
    data_list:list of k array with shape(n_example,2) or two columns(y_trues,y_preds).
    """
    # empty plot
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=18)
    ax.set_xlabel("FPR", fontdict={"fontsize": 18})
    ax.set_ylabel("TPR", fontdict={"fontsize": 18})
    ax.set_title(title if title else "ROC Curve",
                 fontdict={"fontsize": 15}, y=1.05)

    # compute
    rank = {score: index for index, score in enumerate(
        [metrics.roc_auc_score(i[:, 0], i[:, 1]) for i in data_list])}
    for score, index_i in sorted(rank.items(), reverse=True):
        y_true, y_pred = data_list[index_i][:, 0], data_list[index_i][:, 1]
        auc_score = metrics.roc_auc_score(y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        ax.plot(fpr, tpr,
                label=f"{labels[index_i]}:{auc_score:.3f}" if labels else f"{auc_score:.3f}",
                color=colors[index_i] if colors else None,
                linewidth=2.0)
    plt.legend(fontsize=15, shadow=False, framealpha=0)
    plt.savefig(save_file, dpi=600) if save_file else None
    plt.show()


def auprc_curve(y_true, y_pred,save_file=False,label="",title="",color=False,*argvs):
    """
    y_true: true label
    y_pred: pred probility
    
    output: auprc curve
        x-axil :recall  
        y-axil :precision
    """
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #axises
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=26)

    #plot
    AUPRC = round(metrics.average_precision_score(y_true, y_pred), 5)
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred)
    print("x:",recall)
    print("y:",precision )
    print("threshold",threshold)
    if color:
        ax.plot(recall, precision, label=label , color=color, linewidth=2.0)
    else:
        ax.plot(recall, precision, label=label , linewidth=2.0)

    #title and label
    ax.set_xlabel("Recall=TP/(TP+FN)", fontdict={"fontsize": 22})
    ax.set_ylabel("Precision = TP/(TP+FP)", fontdict={"fontsize": 22})
    ax.set_title(title, fontdict={"fontsize": 15}, y=1.05)
    
    return AUPRC




def roc_curve_kfold(data_list, labels=None, title=None, colors=None, save_file=False, alpha=0.1, std_show=False):
    """
    data_list:list of fold_list that contain k array with shape(n_example,2) or two column(y_trues,y_pred)
    """
    # empty plot
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=18)
    ax.set_xlabel("FPR", fontdict={"fontsize": 18})
    ax.set_ylabel("TPR", fontdict={"fontsize": 18})
    ax.set_title(label=title if title else "ROC Curve",
                 fontdict={"fontsize": 15}, y=1.05)
    default_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # (1) rank
    score_index_ranked = sorted(zip([np.mean([metrics.roc_auc_score(
        j[:, 0], j[:, 1])for j in i]) for i in data_list], np.arange(len(data_list))), reverse=True)
    index_ranked = [i[1] for i in score_index_ranked]

    for rank_i in index_ranked:
        fold_list = data_list[rank_i]

        auroc_score_list = []
        fpr_list = []
        tpr_list = []
        for sub_arr in fold_list:
            y_true = sub_arr[:, 0]
            y_pred = sub_arr[:, 1]
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            auroc_score = metrics.roc_auc_score(y_true, y_pred)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auroc_score_list.append(auroc_score)

        x_tick = fpr_list[np.argmax([len(i) for i in fpr_list])]
        y_ticks = np.zeros((len(fold_list), len(x_tick)))
        for i in range(len(fold_list)):
            for idx_j, j in enumerate(x_tick):
                # insert_value(j,fpr_list[i], tpr_list[i])
                y_ticks[i][idx_j] = np.interp(j, fpr_list[i], tpr_list[i])

        # plot
        if std_show:
            if labels:
                ax_label = f"{labels[rank_i]} {np.mean(auroc_score_list):.3f}±{np.std(auroc_score_list):.3f}"
            else:
                ax_label = f"{np.mean(auroc_score_list):.3f}±{np.std(auroc_score_list):.3f}",
        else:
            if labels:
                ax_label = f"{labels[rank_i]} {np.mean(auroc_score_list):.3f}"
            else:
                ax_label = f"{np.mean(auroc_score_list):.3f}"

        ax.plot(x_tick,
                y_ticks.mean(0),
                label=ax_label,
                color=colors[rank_i] if colors else default_color[rank_i])

        ax.fill_between(x_tick,
                        y_ticks.mean(0)+y_ticks.std(0),
                        y_ticks.mean(0)-y_ticks.std(0),
                        facecolor=colors[rank_i] if colors else default_color[rank_i],
                        alpha=alpha)

    plt.legend(fontsize=15, shadow=False, framealpha=0)
    plt.savefig(save_file, dpi=600) if save_file else None
    # plt.show()
    return ax




if __name__ == "__main__":
    y_true = [0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1  ]
    y_pred = [0.0+1e-6, 0.2, 0.4, 0.6, 0.8, 1.0-1e-6, 0.0+1e-6, 0.2, 0.4, 0.6, 0.8, 1.0-1e-6]
    auprc = auprc_curve(y_true,y_pred)
    print(f"auprc:{auprc}")
    plt.show()

