"""
Evaluation taken from AVA and ActivityNet repository
"""
import sys
sys.path.append('../evaluation/ava_evaluation')
import os
import csv
import logging
import numpy as np
import pprint
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpre
from sklearn.preprocessing import label_binarize
import time
from collections import defaultdict
from tqdm import tqdm
from iopath.common.file_io import g_pathmgr

from slowfast.evaluate.ava_evaluation import ( 
    object_detection_evaluation,
    standard_fields
)
#from ava_evaluation import distributed as du

def normalize_bbox(x1, y1, w, h, im_w, im_h):
    x2 = (x1+w)/im_w
    y2 = (y1+h)/im_h
    x1 = x1/im_w
    y1 = y1/im_h
    return x1, y1, x2, y2

def eval_detection(task, coco_anns, preds, img_ann_dict):
    # Transform data to pascal format
    if task == 'actions':
        categories = coco_anns['action_categories']
    else:
        categories = coco_anns['categories']
    num_classes = len(categories)
    print("Formating annotations and preds...")
    groundtruth1 = organize_data_pascal(coco_anns,img_ann_dict,task)
    detections= organize_pred_pascal(groundtruth1[0].keys(),preds,task,num_classes)
    excluded_keys = []
    print("Starting evaluation...")
    results = run_evaluation(categories, groundtruth1, detections, excluded_keys)
    return results["PascalBoxes_Precision/mAP@0.5IOU"]


def read_csv(csv_file, class_whitelist=None, load_score=False):
    """Loads boxes and class labels from a CSV file in the AVA format.
    CSV file format described at https://research.google.com/ava/download.html.
    Args:
      csv_file: A file object.
      class_whitelist: If provided, boxes corresponding to (integer) class labels
        not in this set are skipped.
    Returns:
      boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
      labels: A dictionary mapping each unique image key (string) to a list of
        integer class lables, matching the corresponding box in `boxes`.
      scores: A dictionary mapping each unique image key (string) to a list of
        score values lables, matching the corresponding label in `labels`. If
        scores are not provided in the csv, then they will default to 1.0.
    """
    # This works only with actions
    class_whitelist = set(range(0,18))
    boxes = defaultdict(list)
    labels = {'actions': defaultdict(list), 
              'phases': defaultdict(list),
              'steps': defaultdict(list),
              'tools': defaultdict(list)
              }
    scores = defaultdict(list)
    with g_pathmgr.open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            #! MODIFICAR TAMAÑO PARA NUESTRAS ANOTACIONES
            assert len(row) in [7, 8, 12, 13], "Wrong number of columns: " + row
            image_key = "%s/%05d.jpg" % (row[0], int(row[1]))
            x1, y1, x2, y2 = [float(n) for n in row[3:7]]
            # TODO: fix eval for multiple actions and tasks
            action_id1 = int(row[7]) 
            action_id2 = int(row[8]) 
            action_id3 = int(row[9])
            action_list = [action_id1, action_id2, action_id3]
            action_list = [act for act in action_list if act != -1] # ¿Hacemos esto?
            step_id = int(row[11]) #Verificar qué va primero
            phase_id = int(row[12].split('.')[0])
            tools_id = int(row[10]) 
            actions_list = []
            # TODO: Hacer algo más elegante (?)
            # breakpoint()
            for action_id in action_list:
                if class_whitelist and action_id not in class_whitelist:
                    continue
                else:
                    actions_list.append(action_id)
            if len(actions_list) > 0:
                score = 1.0
                if load_score:
                    score = float(row[13])
                for action in actions_list: #Tiene que haber mismo número de cajas que de labels
                    new_box = [y1, x1, y2, x2]
                    boxes[image_key].extend([new_box])
                    labels['actions'][image_key].extend([action])
                    # boxes[image_key].append([y1, x1, y2, x2])

                    labels['tools'][image_key].extend([tools_id]) 
                    # scores[image_key].append(score)
                #     labels['actions'][image_key].append(action)

                # labels['phases'][image_key].append(phase_id)
                # labels['steps'][image_key].append(step_id)
            
                # labels[image_key].append(actions_list)
                
            else:

                continue
    scores = []
    return boxes, labels['tools'], scores


def organize_data_pascal(coco_anns, img_ann_dict, task):
    '''
        bboxes for groundtruth are in [x1,y1,w,h]
    '''
    excluded_keys = set()

    bboxes = defaultdict(list)
    labels = defaultdict(list)

    for img_name, img_idx in tqdm(img_ann_dict.items()): 
        img = [img for img in coco_anns["images"] if img["file_name"] == img_name][0]
        im_w = img["width"]
        im_h = img["height"]
        if len(img_idx) == 0:
            continue
        for idx in img_idx:
            if task == 'actions':
                try:
                    lbl = coco_anns['annotations'][idx]['actions']
                except:
                    breakpoint()
            else:
                lbl = [coco_anns['annotations'][idx]['category_id']]
                
            x1, y1, w, h = coco_anns['annotations'][idx]['bbox']
            x1, y1, x2, y2 = normalize_bbox(x1, y1, w, h, im_w, im_h)
            new_bbox = [y1,x1,y2,x2]
            
            for a_idx, act in enumerate(lbl):
                # Each new action, bbox, img has a new key
                new_key = img_name
                # new_key = '{}_{}_{}'.format(img_name, idx, a_idx)
                bboxes[new_key].extend([new_bbox])
                labels[new_key].extend([act])
    groundtruth = [bboxes, labels, []]
    return groundtruth

def organize_pred_pascal(gt_keys, preds, task, num_classes):
    '''
        bboxes for preds are in format [x1,y1,x2,y2]
    '''

    pred_bboxes = defaultdict(list)
    pred_scores = defaultdict(list)
    pred_labels = defaultdict(list)
    pred_keys = list(preds.keys())
    for new_key in tqdm(gt_keys): 
        img_name = new_key.split('_')[0]
        if img_name not in pred_keys:
            continue
        pred_image = preds[img_name]["bboxes"]
        
        if len(pred_image) == 0:
            continue

        for this_box in pred_image:
            box, prob_task = this_box['bbox'], this_box['prob_'+task]
            x1, y1, x2, y2 = box
            new_box = [y1, x1, y2, x2]
            pred_bboxes[new_key].extend([new_box for _ in range(num_classes)])
            pred_scores[new_key].extend(prob_task)
            pred_labels[new_key].extend(list(range(1, num_classes + 1)))
   
    detection = [pred_bboxes, pred_labels, pred_scores]

    return detection

def convert(gt):
    boxes, labels, scores = gt
    classes = 16
    out_scores = defaultdict(list)
    out_labels = defaultdict(list)
    out_boxes = defaultdict(list)
    n_labels = list(range(1,classes+1))
    for key in boxes.keys():
        for box, lab in zip(boxes[key], labels[key]):
            out_boxes[key].extend([box]*classes)
            out_labels[key].extend(n_labels)
            score = [0]*classes
            score[lab-1] = 1
            out_scores[key].extend(score)

    return [out_boxes, out_labels, out_scores]

def run_evaluation(
    categories, groundtruth, detections, excluded_keys, verbose=True
):
    """AVA evaluation main logic."""

    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories
    )

    boxes, labels, _ = groundtruth
    gt_keys = []
    pred_keys = []

    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(
                (
                    "Found excluded timestamp in ground truth: %s. "
                    "It will be ignored."
                ),
                image_key,
            )
            continue
        
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key,
            {
                standard_fields.InputDataFields.groundtruth_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.InputDataFields.groundtruth_classes: np.array(
                    labels[image_key], dtype=int #Tiene que haber mismo número de cajas que de labels
                ),
                standard_fields.InputDataFields.groundtruth_difficult: np.zeros(
                    len(boxes[image_key]), dtype=bool
                ),
            },
        )

        gt_keys.append(image_key)

    boxes, labels, scores = detections
    # boxes1, labels1, scores1 = convert(groundtruth)
    # # breakpoint()
    # l1 = [1 for key in boxes.keys() if boxes[key] != boxes1[key]]
    # l2 = [1 for key in labels.keys() if labels[key] != labels1[key]]
    # l3 = [1 for key in boxes.keys() if scores[key] != scores1[key]]


    # import pdb; pdb.set_trace()

    for image_key in tqdm(boxes):
        if image_key in excluded_keys:
            logging.info(
                (
                    "Found excluded timestamp in detections: %s. "
                    "It will be ignored."
                ),
                image_key,
            )
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key,
            {
                standard_fields.DetectionResultFields.detection_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.DetectionResultFields.detection_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.DetectionResultFields.detection_scores: np.array(
                    scores[image_key], dtype=float
                ),
            },
        )

        pred_keys.append(image_key)
    print("Calculating metric...")
    metrics = pascal_evaluator.evaluate()

    # if du.is_master_proc():
    #     pprint.pprint(metrics, indent=2)
    return metrics