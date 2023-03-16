#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

from genericpath import exists
import random
import numpy as np
import shutil
import os
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats


import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc

from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import EpochTimer, SurgeryMeter

logger = logging.get_logger(__name__)

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py

    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    complete_tasks = cfg.TASKS.TASKS
    complete_loss_funs = cfg.TASKS.LOSS_FUNC
    
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if isinstance(val[i], (list,)):
                            val[i] = val[i][0].cuda(non_blocking=True)
                        else:
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            for key, val in labels.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if isinstance(val[i], (list,)):
                            val[i] = val[i][0].cuda(non_blocking=True)
                        else:
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    labels[key] = val.cuda(non_blocking=True)
                    
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            faster_ftrs = meta["faster_features"] if cfg.FASTER.ENABLE else None
            boxes_mask = meta["boxes_mask"] if cfg.FASTER.ENABLE else None
            preds = model(inputs, meta["boxes"], faster_ftrs, boxes_mask)
            keep_box = meta["keep_box"]
            # Explicitly declare reduction to mean and compute the loss for each task.
            loss = []
            for idx, task in enumerate(complete_tasks):
                if task == 'actions':
                    loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                    loss.append(loss_fun(preds[task][0][keep_box], labels[task][keep_box]))    
                elif task == 'tools':
                    loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                    loss.append(loss_fun(preds[task][0][keep_box], labels[task][keep_box].long()))    
                else:
                    loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                    indexes = np.unique(meta["boxes"][:,0].cpu(),return_index=True)[1]
                    loss.append(loss_fun(preds[task][0], labels[task][indexes].long()))

        if len(complete_tasks) >1:
            
            final_loss = losses.compute_weighted_loss(loss, cfg.TASKS.LOSS_WEIGHTS)
        else:
            final_loss = loss[0]
            
        # check Nan Loss.
        misc.check_nan_losses(final_loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(final_loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.NUM_GPUS > 1:
            final_loss = du.all_reduce([final_loss])[0]
        final_loss = final_loss.item()

        # Update and log stats.
        train_meter.update_stats(None, None, None, None, None, final_loss, loss, lr)
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    complete_tasks = cfg.TASKS.TASKS
    
    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            
            for key, val in labels.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if isinstance(val[i], (list,)):
                            val[i] = val[i][0].cuda(non_blocking=True)
                        else:
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    labels[key] = val.cuda(non_blocking=True)
                    
        val_meter.data_toc()
        faster_ftrs = meta["faster_features"] if cfg.FASTER.ENABLE else None
        boxes_mask = meta["boxes_mask"] if cfg.FASTER.ENABLE else None
        preds = model(inputs, meta["boxes"], faster_ftrs, boxes_mask)
        keep_box = meta["keep_box"]
        ori_boxes = meta["ori_boxes"]
        metadata = meta["metadata"]
        image_names = meta["img_names"]
        if cfg.NUM_GPUS:
            preds = {task: preds[task][0].cpu() for task in complete_tasks}
            ori_boxes = ori_boxes.cpu()
            metadata = metadata.cpu()
            keep_box = keep_box.cpu()
            image_names = image_names.cpu()
            if cfg.NUM_GPUS > 1:
                preds_gather = {}
                for pred in preds:
                    preds_gather[pred] = torch.cat(du.all_gather_unaligned(preds[pred]), dim=0)
                preds = preds_gather.copy()
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)
                image_names = torch.cat(du.all_gather_unaligned(image_names), dim=0)
                keep_box = torch.cat(du.all_gather_unaligned(keep_box), dim=0)

        val_meter.iter_toc()
        epoch_names_detect, epoch_bboxes = [], []
        for image_name, batch_box in zip(image_names[keep_box], ori_boxes[keep_box].cpu().tolist()):
            epoch_names_detect.append(''.join(map(chr,image_name.cpu().tolist())))
            epoch_bboxes.append([batch_box[j] for j in [1, 2, 3, 4]])

        # Images names phases/steps
        epoch_names = []
        for image_name in image_names:
            epoch_names.append(''.join(map(chr,image_name.cpu().tolist())))   
        
        epoch_names = list(np.unique(epoch_names))   
        
        # Update and log stats.
        val_meter.update_stats(preds, keep_box, epoch_bboxes, epoch_names_detect, epoch_names)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    val_meter.log_epoch_stats(cur_epoch)

    if cfg.NUM_GPUS > 1:
        if du.is_master_proc():
            task_map, mean_map, out_files = val_meter.finalize_metrics()
        else:
            task_map, mean_map, out_files =  [0, 0, 0]
        torch.distributed.barrier()
    else:
        task_map, mean_map, out_files = val_meter.finalize_metrics()
    val_meter.reset()

    return task_map, mean_map, out_files


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = SurgeryMeter(len(train_loader), cfg, mode="train")
    val_meter = SurgeryMeter(len(val_loader), cfg, mode="val")

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # TODO Si no corre, quitarlo
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
            
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )
    
    # Create meters.
    train_meter = SurgeryMeter(len(train_loader), cfg, mode="train")
    val_meter = SurgeryMeter(len(val_loader), cfg, mode="val")

    # Perform final test
    if cfg.TEST.ENABLE:
        logger.info("Evaluating epoch: {}".format(start_epoch + 1))
        map_task, mean_map, out_files = eval_epoch(val_loader, model, val_meter, start_epoch, cfg)
        return
    else:
        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))
        
    # Stats for saving checkpoint:
    complete_tasks = cfg.TASKS.TASKS
    best_task_map = {task: 0 for task in complete_tasks}
    best_mean_map = 0
    epoch_timer = EpochTimer()
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        
        del_fil = os.path.join(cfg.OUTPUT_DIR,'checkpoints', 'checkpoint_epoch_{0:05d}.pyth'.format(cur_epoch-1))
        if os.path.exists(del_fil):
            os.remove(del_fil)
            
        # Evaluate the model on validation set.
        if is_eval_epoch:
            map_task, mean_map, out_files = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            if (cfg.NUM_GPUS > 1 and du.is_master_proc()) or cfg.NUM_GPUS == 1:
                main_path = os.path.split(list(out_files.values())[0])[0]
                fold = main_path.split('/')[-1]
                best_preds_path = main_path.replace(fold, fold+'/best_predictions')
                if not os.path.exists(best_preds_path):
                    os.makedirs(best_preds_path)
                # Save best results
                if mean_map > best_mean_map:
                    best_mean_map = mean_map
                    logger.info("Best mean map at epoch {}".format(cur_epoch))
                    cu.save_best_checkpoint(
                        cfg.OUTPUT_DIR,
                        model,
                        optimizer,
                        'mean',
                        cfg,
                        scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
                    for task in complete_tasks:
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best_all') )
                        shutil.copyfile(out_files[task], copy_path)
                
                for task in complete_tasks:
                    if map_task[task] > best_task_map[task]:
                        best_task_map[task] = map_task[task]
                        logger.info("Best {} map at epoch {}".format(task, cur_epoch))
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best') )
                        shutil.copyfile(out_files[task], copy_path)
                        cu.save_best_checkpoint(
                            cfg.OUTPUT_DIR,
                            model,
                            optimizer,
                            task,
                            cfg,
                            scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
    cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )

