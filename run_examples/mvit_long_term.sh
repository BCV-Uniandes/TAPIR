# Experiment setup
FOLD="1" # Fold of the cross-validation split.
EXP_NAME="TAPIR_Phases"
TASK="PHASES" # Long term tasks "PHASES" for the phases recognition or "STEPS" for the steps recognition task
CHECKPOINT="PSI-AVA/TAPIR_trained_models/PHASES/Fold"$FOLD"/checkpoint_best_phases.pyth"  # Path to the model weights of the pretrained model

#-------------------------
DATA_VER="psi-ava_extended"
EXPERIMENT_NAME=$EXP_NAME"/Fold"$FOLD
CONFIG_PATH="configs/MVIT_"$TASK".yaml"
MAP_FILE="surgery_"$TASK"_list.pbtxt"
FRAME_DIR="outputs/PSIAVA/keyframes" # Path to the organized keyframes
OUTPUT_DIR="outputs/log/"$TASK"/"$EXPERIMENT_NAME
FRAME_LIST="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/frame_lists"
ANNOT_DIR="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/annotations"
COCO_ANN_PATH="outputs/data_annotations/"$DATA_VER"/fold"$FOLD"/coco_anns/val_coco_anns_DB1_v3_1s.json"
TYPE="pytorch"
#-------------------------
# Run experiment

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
AVA.TRAIN_USE_COLOR_AUGMENTATION True \
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
OUTPUT_DIR $OUTPUT_DIR 