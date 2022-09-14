import os
import os.path as osp
import json
import numpy as np
import sklearn
import argparse

from slowfast.evaluate.classification_eval import eval_classification
from slowfast.evaluate.detection_eval import eval_detection


def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def save_json(data, json_file, indent=4):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)


def get_img_ann_dict(coco_anns):
    img_ann_dict = {}
    for img in coco_anns["images"]:
        img_name = img["file_name"]
        img_ann_dict[img_name] = [idx for idx, ann in enumerate(coco_anns["annotations"]) if ann["image_name"] == img_name]
    return img_ann_dict

def eval_task(task, coco_anns, preds, visualization):
    img_ann_dict = get_img_ann_dict(coco_anns)
    if task == 'phases' or task == 'steps':
        mAP, mP, mR = eval_classification(task, coco_anns, preds, visualization)
    elif task == 'tools' or task == 'actions':
        mAP = eval_detection(task, coco_anns, preds, img_ann_dict)
    else:
        raise('Unknown task')
    return mAP

def main(coco_ann_path, pred_path, tasks=['phases', 'steps'], visualization=False):
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds = load_json(pred_path)
    all_mAP = {}
    for task in tasks:
        task_mAP = eval_task(task, coco_anns, preds, visualization)
        all_mAP[task] = task_mAP
        print('{} task mAP: {}'.format(task, task_mAP))
    overall_mAP = np.mean(list(all_mAP.values()))
    print('Overall mAP: {}'.format(overall_mAP))
    return overall_mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation parser')
    parser.add_argument('--coco-ann-path', default=None,
                        type=str, help='path to coco style anotations')
    parser.add_argument('--coco-pred-path', default=None,
                        type=str, help='path to predictions')
    parser.add_argument('--tasks', nargs='+', help='tasks to be evaluated',
                        required=True, default=None)
    parser.add_argument('--visualization', default=False, action='store_true')

    args = parser.parse_args()
    print(args)
    main(args.coco_ann_path, args.coco_pred_path, args.tasks, args.visualization)