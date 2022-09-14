import os
import json
import numpy as np
from slowfast.evaluate.main_eval import load_json


def normalize_bbox(x1, y1, w, h, im_w, im_h):
    x2 = (x1+w)/im_w
    y2 = (y1+h)/im_h
    x1 = x1/im_w
    y1 = y1/im_h
    return x1, y1, x2, y2

def main(coco_ann_path, task):
    coco_anns = load_json(coco_ann_path)
    if task == 'actions':
        classes = coco_anns["action_categories".format(task)]
    elif task == 'tools':
        classes = coco_anns["categories".format(task)]
    else:
        classes = coco_anns["{}_categories".format(task)]
    annotations = coco_anns['annotations']
    save_json = {}
    if task in ['actions', 'tools']:
        for ann in annotations:
            name = ann['image_name']
            task_key_name = 'prob_'+task
            if task == 'actions':
                pred = list(np.zeros(len(classes)))
                for act in ann['actions']:
                    idx = int(act)
                    pred[idx-1] = 1
            #     
            else:

                pred = list(np.zeros(len(classes)))
                idx = int(ann['category_id'])
                pred[idx-1] = 1
            task_key_name = 'prob_'+task
            x1, y1, w, h =  ann['bbox']
            im_w, im_h =  1280, 800
           
            box = normalize_bbox(x1, y1, w, h, im_w, im_h)
            save_dict = {'bbox': box, task_key_name: pred}
            if name not in save_json.keys():
                save_json[name] = {'bboxes': [save_dict]}
            else:
                save_json[name]['bboxes'].append(save_dict)
    else:
        for ann in annotations:
            name = ann['image_name']
            task_key_name = 'prob_'+task
            pred = list(np.zeros(len(classes)))
            idx = int(ann[task[:-1]])
            pred[idx] = 1
            task_key_name = 'prob_'+task
            save_dict = {task_key_name: pred}
            save_json[name] = save_dict
    breakpoint()
    path_prediction = os.path.join('.',f'best_epoch_preds_{task}.json')
    with open(path_prediction, "w") as outfile:  
        json.dump(save_json, outfile)
   


if __name__ == "__main__":
    coco_ann_path = '/media/SSD6/MICCAI2021/data/SantaFeDataset/annotations/ava_surgery_minifinal_extra_cands/fold1/coco_anns/train_coco_anns_v2.json'
    tasks = ['steps', 'phases', 'actions'] #, 'tools']
    for task in tasks[::-1]:
        main(coco_ann_path, task)
        break
