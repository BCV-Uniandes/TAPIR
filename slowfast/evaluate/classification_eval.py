import os
import os.path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.backends.backend_pdf
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score


def eval_classification(task, coco_anns, preds, visualization):
    
    if task == 'phases':
        classes = coco_anns["phase_categories".format(task)]
    elif task == 'steps':
        classes = coco_anns["step_categories".format(task)]
    else:
        classes = coco_anns["{}_categories".format(task)]
    num_classes = len(classes)
    bin_labels = np.zeros((len(coco_anns["annotations"]), num_classes))
    bin_preds = np.zeros((len(coco_anns["annotations"]), num_classes))
    evaluated_frames = []
    for idx, ann in enumerate(coco_anns["annotations"]):
        ann_class = int(ann[task[:-1]])
        bin_labels[idx, :] = label_binarize([ann_class], classes=list(range(0, num_classes)))
        # TODO Quitar cuando se hable de los datos.
        if  ann["image_name"] in preds.keys():
            these_probs = preds[ann["image_name"]]['prob_{}'.format(task)]
            if len(these_probs) == 0:
                print("Prediction not found for image {}".format(ann["image_name"]))
                these_probs = np.zeros((1, num_classes))
            else:
                evaluated_frames.append(idx)
            bin_preds[idx, :] = these_probs
        else:
            print("Image {} not found in predictions lists".format(ann["image_name"]))
            these_probs = np.zeros((1, num_classes))
            bin_preds[idx, :] = these_probs
            
    bin_labels = bin_labels[evaluated_frames]
    bin_preds = bin_preds[evaluated_frames]
    
    if visualization:
        visualize_results(bin_labels,bin_preds,task)
    precision = {}
    recall = {}
    threshs = {}
    ap = {}
    for c in range(0, num_classes):
        precision[c], recall[c], threshs[c] = precision_recall_curve(bin_labels[:, c], bin_preds[:, c])

        ap[c] = average_precision_score(bin_labels[:, c], bin_preds[:, c])
    mAP = np.nanmean(list(ap.values()))
    # TODO: save PR curve plot with all curves
    return mAP, precision, recall

def visualize_results(bin_labels,bin_preds,task):
    
    path_save = 'Visualization_'+task+'.jpg'
    
    dict_color={0:'xkcd:lightblue',1:'b',2:'g',3:'r',4:'c',5:'m',6:'y',7:'k',8:'tab:orange',9:'tab:brown',10:'tab:pink',11:'tab:gray',12:'lime',13:'tab:blue',14:'xkcd:gold',15:'xkcd:maroon',16:'xkcd:khaki',17:'xkcd:indigo',18:'xkcd:magenta',19:'xkcd:aqua',20:'xkcd:aqua'}
    labels = pd.Series()
    preds = pd.Series()
    labels_colors = pd.Series()
    preds_colors = pd.Series()
    wrong_predictions = 0
    frame_total = pd.Series()
    count = 0

    for idx in range(len(bin_labels)):
        label_category = int(np.nonzero(bin_labels[idx,:])[0])
        prediction_category = int(np.argmax(bin_preds[idx,:]))
        if label_category != prediction_category:
            wrong_predictions+=1
        labels = labels.append(pd.Series(label_category))
        preds = preds.append(pd.Series(prediction_category))
        frame_total = frame_total.append(pd.Series(count))
        count+=1
    
    labels_colors = np.array(labels.replace(to_replace=dict_color))
    preds_colors = np.array(preds.replace(to_replace=dict_color))
    labels = np.array(labels)
    preds = np.array(preds)

    #PLOT
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,sharex=True)
    plt.subplots_adjust(hspace=0.8)
    fig.suptitle('WrongPredictions {}/{}'.format(wrong_predictions,count), fontsize=16)
    #ax1.vlines(frames_p,ymin=0,ymax=10,colors=colors_p,linewidths=0.5)
    ax1.vlines(frame_total,ymin=0,ymax=10,colors=preds_colors,linewidths=0.5)
    ax1.set_title('Predictions')
    ax2.vlines(frame_total,ymin=0,ymax=10,colors=labels_colors,linewidths=0.5)
    ax2.set_title('Annotations')

    plt.savefig(path_save)

