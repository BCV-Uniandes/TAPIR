#!/bin/bash
FOLD='fold1'
TASKS=('tools')
COCO_ANN_PATH='/media/SSD6/MICCAI2021/data/SantaFeDataset/annotations/ava_surgery_minifinal_extra_cands/'$FOLD'/coco_anns/val_coco_anns_v2.json'
COCO_PRED_PATH='/media/SSD0/MICCAI2021/Experiments/Slowfast/Joint_Model/All_Tasks_FromSteps/Fold1/EVAL_03/best_epoch_preds_'$TASKS'.json'

python main_eval.py --coco-ann-path $COCO_ANN_PATH \
--coco-pred-path $COCO_PRED_PATH --tasks $TASKS
# add --visualization for vis
