# Experiment setup
FOLD="2" # Fold of the cross-validation split.
EXP_NAME="TAPIR_Instruments"
TASK="TOOLS" # Short term tasks "TOOLS" for the instrument detection task or "ACTIONS" for the atomic action recognition task
CHECKPOINT="PSI-AVA/TAPIR_trained_models/INSTRUMENTS/Fold"$FOLD"/checkpoint_best_tools.pyth"  # Path to the model weights of the pretrained model

#-------------------------
DATA_VER="psi-ava"
EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
CONFIG_PATH="configs/MVIT_"$TASK".yaml"
MAP_FILE="surgery_"$TASK"_list.pbtxt"
FRAME_DIR="outputs/PSIAVA/keyframes/" # Path to the organized keyframes
OUTPUT_DIR="outputs/log/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
ANNOT_DIR="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/annotations"
COCO_ANN_PATH="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns_v3_35s.json"
FF_TRAIN="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/train/bbox_features.pth" # Path to the intrument bounding boxes and features in the training set
FF_VAL="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/val/bbox_features.pth" # Path to the intrument bounding boxes and features in the validating set

TYPE="pytorch"
#-------------------------
# Run experiment

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1 python tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TRAIN.CHECKPOINT_TYPE $TYPE \
TEST.ENABLE False \
AVA.FRAME_DIR $FRAME_DIR \
AVA.FRAME_LIST_DIR $FRAME_LIST \
AVA.ANNOTATION_DIR $ANNOT_DIR \
AVA.LABEL_MAP_FILE $MAP_FILE \
AVA.COCO_ANN_DIR $COCO_ANN_PATH \
BN.NUM_BATCHES_PRECISE 72 \
FASTER.FEATURES_TRAIN $FF_TRAIN \
FASTER.FEATURES_VAL $FF_VAL \
OUTPUT_DIR $OUTPUT_DIR