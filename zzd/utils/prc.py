from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


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




def pr_curve_kfold(data_list, 
                    labels=None, 
                    title=None, 
                    colors=None, 
                    save_file=False, 
                    alpha=0.0, 
                    std_show=False,
                    bbox_to_anchor=None):
    """
    data_list:list of k folds data. while data is an array with shape(n_example,2).
    """
    # empty plot
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=18)
    ax.set_xlabel("Recall", fontdict={"fontsize": 18})
    ax.set_ylabel("Precision", fontdict={"fontsize": 18})
    ax.set_title(label=title if title else "PR Curve",
                 fontdict={"fontsize": 15}, y=1.05)
    default_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # compute auprc score
    def auprc(df):
        return np.round(metrics.average_precision_score(df[:,0], df[:,1]), 5)
    
    # (1) rank
    score_index_ranked = sorted(zip([np.mean([auprc(j) for j in i])
                                for i in data_list], np.arange(len(data_list))), reverse=True)
    index_ranked = [i[1] for i in score_index_ranked]

    # (3) compute y lim of precision
    for rank_i in index_ranked:

        fold_list = data_list[rank_i]
        auprc_score_list = []
        precision_list = []
        recall_list = []

        for sub_arr in fold_list:
            y_true = sub_arr[:, 0]
            y_pred = sub_arr[:, 1]
            precision, recall, _ = metrics.precision_recall_curve( y_true, y_pred)
            auprc_score = metrics.average_precision_score(y_true, y_pred)
            
            recall = np.flip(recall)
            precision = np.flip(precision)
            
            precision_list.append(precision)
            recall_list.append(recall)
            auprc_score_list.append(auprc_score)

        #设置x_tick
        x_tick = sorted(set([ _ for recall in recall_list for _ in recall]))

        
        #补全所有y_tick
        y_ticks = np.zeros((len(fold_list), len(x_tick)))
        
        for i_fold in range(len(fold_list)):
            for idx_j, j in enumerate(x_tick):
                    y_ticks[i_fold][idx_j] = np.interp(j, recall_list[i_fold], precision_list[i_fold])
        # plot
        if std_show:
            if labels:
                ax_label = f"{labels[rank_i]} {np.mean(auprc_score_list):.3f}±{np.std(auprc_score_list):.3f}"
            else:
                ax_label = f"{np.mean(auprc_score_list):.3f}±{np.std(auprc_score_list):.3f}",
        else:
            if labels:
                ax_label = f"{labels[rank_i]} {np.mean(auprc_score_list):.3f}"
            else:
                ax_label = f"{np.mean(auprc_score_list):.3f}"

        ax.plot(x_tick,
                y_ticks.mean(0),
                label=ax_label,
                color=colors[rank_i-1] if colors else default_color[rank_i])

        ax.fill_between(x_tick,
                        y_ticks.mean(0)+y_ticks.std(0),
                        y_ticks.mean(0)-y_ticks.std(0),
                        facecolor=colors[rank_i -
                                         1] if colors else default_color[rank_i],
                        alpha=alpha)

    plt.legend(fontsize=13, shadow=False, framealpha=0, bbox_to_anchor=bbox_to_anchor)
    plt.savefig(save_file, dpi=600) if save_file else None
    # plt.show()
    return ax


# if __name__ == "__main__":
#     y_true = [0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1  ]
#     y_pred = [0.0+1e-6, 0.2, 0.4, 0.6, 0.8, 1.0-1e-6, 0.0+1e-6, 0.2, 0.4, 0.6, 0.8, 1.0-1e-6]
#     auprc = auprc_curve(y_true,y_pred)
#     print(f"auprc:{auprc}")
#     plt.show()

