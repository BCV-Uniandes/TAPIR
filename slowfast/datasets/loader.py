#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
import numpy as np
from functools import partial
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from . import utils as utils
from .build import build_dataset


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    breakpoint()
    inputs = [item for sublist in inputs for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key in ["ori_boxes","boxes"]:
            # Append idx info to the bboxes before concatenating them.
            try:
                bboxes = [
                    np.concatenate(
                        [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                    )
                    for i in range(len(data))
                ]
            except:
                print(key)
                import pdb; pdb.set_trace()
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
            
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                np.array(list(itertools.chain(*data)))
            ).view(-1, 2)
        elif key == "img_names":
            # String to ascii to return in tensor mode
            mod_data = [list(map(ord, i[0])) for i in itertools.chain(*data)]
            collated_extra_data[key] = torch.tensor(mod_data)
        elif key == "extra_labels":
            collated_extra_data[key] = torch.tensor(
                    np.array(list(itertools.chain(*data)))
                ).view(-1, data[0].shape[1])
        elif key == "keep_box":
            collated_extra_data[key] = torch.tensor(
                np.array(list(itertools.chain(*data)))
            )
        elif key == "boxes_mask":
            collated_extra_data[key] = default_collate(data).bool()
        else:
            collated_extra_data[key] = default_collate(data).float()
    
    collated_labels = {}
    for key in labels[0].keys():
        if key == 'actions':
            data = [torch.tensor(d[key]).float() for d in labels]
        else:
            data = [torch.tensor(d[key]) for d in labels]
        collated_labels[key] = torch.cat(data,dim=0)
        
    return inputs, collated_labels, video_idx, collated_extra_data, 


def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    else:
        # Create a sampler for multi-process training
        sampler = utils.create_sampler(dataset, shuffle, cfg)
        # Create a loader
        if cfg.DETECTION.ENABLE:
            collate_func = detection_collate
        elif cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
            collate_func = partial(
                multiple_samples_collate, fold="imagenet" in dataset_name
            )
        else:
            collate_func = None

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=collate_func,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    if (
        loader._dataset_kind
        == torch.utils.data.dataloader._DatasetKind.Iterable
    ):
        if hasattr(loader.dataset, "sampler"):
            sampler = loader.dataset.sampler
        else:
            raise RuntimeError(
                "Unknown sampler for IterableDataset when shuffling dataset"
            )
    else:
        sampler = (
            loader.sampler
        )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
