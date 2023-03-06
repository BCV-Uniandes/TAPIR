#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from cv2 import BOWKMeansTrainer
import torch
import logging
from collections import defaultdict

from slowfast.utils.env import pathmgr

logger = logging.getLogger(__name__)

FPS = 30

# All frames in PSI-AVA data are valid to fit TAPIR
AVA_VALID_FRAMES = range(0, 1000000)


def load_features_boxes(cfg):
    """
    Load boxes features from faster cnn trained model. Used for initialization
    in actions and tools detection.

    Args:
        cfg (CfgNode): config.

    Returns:
        features (tensor): a tensor of faster weights.
    """

    features1 = torch.load(cfg.FASTER.FEATURES_TRAIN)["features"]
    features2 = torch.load(cfg.FASTER.FEATURES_VAL)["features"]
    features = features1 + features2

    return features


def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    list_filenames = [
        os.path.join(cfg.AVA.FRAME_LIST_DIR, filename)
        for filename in (cfg.AVA.TRAIN_LISTS if is_train else cfg.AVA.TEST_LISTS)
    ]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames:
        with pathmgr.open(list_filename, "r") as f:
            # f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path
                assert len(row) == 4
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]
                image_paths[data_key].append(row[3])

    image_paths = [image_paths[i] for i in range(len(image_paths))]
    logger.info("Finished loading image paths from: %s" % ", ".join(list_filenames))

    return image_paths, video_idx_to_name


def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    gt_lists = cfg.AVA.TRAIN_GT_BOX_LISTS if mode == "train" else []
    pred_lists = (
        cfg.AVA.TRAIN_PREDICT_BOX_LISTS
        if mode == "train"
        else cfg.AVA.TEST_PREDICT_BOX_LISTS
    )
    ann_filenames = [
        os.path.join(cfg.AVA.ANNOTATION_DIR, filename)
        for filename in gt_lists + pred_lists
    ]
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(pred_lists)

    detect_thresh = cfg.AVA.DETECTION_SCORE_THRESH
    # Only select frame_sec % 4 = 0 samples for validation if not
    # set FULL_TEST_ON_VAL.
    boxes_sample_rate = 4 if mode == "val" and not cfg.AVA.FULL_TEST_ON_VAL else 1

    all_boxes, count, unique_box_count = parse_bboxes_file(
        ann_filenames=ann_filenames,
        ann_is_gt_box=ann_is_gt_box,
        detect_thresh=detect_thresh,
        boxes_sample_rate=boxes_sample_rate,
    )
    logger.info("Finished loading annotations from: %s" % ", ".join(ann_filenames))
    logger.info("Detection threshold: {}".format(detect_thresh))
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)

    return all_boxes


def get_keyframe_data(boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """

        # PSI-AVA for TAPIR was sampled a frame per second
        return int(sec)

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx].keys():
            sec_idx = list(boxes_and_labels[video_idx].keys()).index(sec)
            if sec not in AVA_VALID_FRAMES:
                continue

            keyframe_indices.append((video_idx, sec_idx, sec, sec_to_frame(sec)))
            keyframe_boxes_and_labels[video_idx].append(
                boxes_and_labels[video_idx][sec]
            )
            count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count


def parse_bboxes_file(
    ann_filenames, ann_is_gt_box, detect_thresh, boxes_sample_rate=1,
):
    """
    Parse PSI-AVA bounding boxes files.
    Args:
        ann_filenames (list of str(s)): a list of PSI-AVA bounding boxes annotation files.
        ann_is_gt_box (list of bools): a list of boolean to indicate whether the corresponding
            ann_file is ground-truth. `ann_is_gt_box[i]` correspond to `ann_filenames[i]`.
        detect_thresh (float): threshold for accepting predicted boxes, range [0, 1].
        boxes_sample_rate (int): sample rate for test bounding boxes. Get 1 every `boxes_sample_rate`.
    """
    all_boxes = {}
    count = 0
    unique_box_count = 0

    ################################################################
    ## ADD THIS BECAUSE OF INCOMPLETE DATASET
    AVA_VALID_FRAMES = {}
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with pathmgr.open(filename, "r") as f:
            for line in f:
                row = line.strip().split(",")
                # When we use predicted boxes to train/eval, we need to
                # ignore the boxes whose scores are below the threshold.
                video_name, frame_sec = row[0], int(row[1])

                if video_name not in AVA_VALID_FRAMES.keys():
                    AVA_VALID_FRAMES[video_name] = [frame_sec]
                else:
                    if frame_sec not in AVA_VALID_FRAMES[video_name]:
                            AVA_VALID_FRAMES[video_name].append(frame_sec)

    #################################################################

    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        with pathmgr.open(filename, "r") as f:
            for line in f:
                row = line.strip().split(",")
                # When we use predicted boxes to train/eval, we need to
                # save as empty the boxes whose scores are below the threshold.
                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                box_key = ",".join(row[3:7])
                box = list(map(float, row[3:7]))

                if not is_gt_box:
                    score = float(row[13])
                    if score < detect_thresh:
                        box = []
                video_name, frame_sec = row[0], int(row[1])
                if frame_sec % boxes_sample_rate != 0:
                    continue
                # TODO: Verificar esto
                # CHANGE WHEN DATABASE IS COMPLETED
                if frame_sec not in AVA_VALID_FRAMES[video_name]:
                    continue

                # We have multiple labels for Atomic Action Detection:
                # Action 1, Action 2, Action 3
                label = []
                for ind in range(7, 10):
                    label.append(-1 if row[ind] == "" else int(row[ind]))

                if video_name not in all_boxes:
                    all_boxes[video_name] = {}
                    for sec in AVA_VALID_FRAMES[video_name]:
                        all_boxes[video_name][sec] = {}

                # all_boxes is a list of 4 elements: boxes, actions anns,
                # instrument ann, phases and steps anns.
                if box_key not in all_boxes[video_name][frame_sec]:
                    all_boxes[video_name][frame_sec][box_key] = [box, [], [], []]
                    unique_box_count += 1

                # Add actions labels
                all_boxes[video_name][frame_sec][box_key][1].extend(label)
                
                # Modify instrument label
                label_inst = int(row[10]) - 1
                # TODO Esto ya no debería pasar. Revisar y quitar
                # -----------------------------------------------
                # Change last label to float
                if float(row[11]) == -1:
                    print('-1 encontrado en phases')
                    row[11] = 0
                if float(row[12]) == -1:
                    row[12] = 0
                # -----------------------------------------------
                
                # Add instrument label
                all_boxes[video_name][frame_sec][box_key][2].append(label_inst)

                # Add phases and steps labels
                all_boxes[video_name][frame_sec][box_key][3].extend(
                        list(map(int, row[11:13]))
                    )

                count += 1

    for video_name in all_boxes.keys():
        for frame_sec in all_boxes[video_name].keys():
            # Save in format of a list of [box_i, box_i_labels].
            all_boxes[video_name][frame_sec] = list(
                all_boxes[video_name][frame_sec].values()
            )

    return all_boxes, count, unique_box_count
