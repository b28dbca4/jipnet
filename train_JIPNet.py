"""
Description:
Author: Xiongjun Guan
Date: 2025-04-25 15:16:19
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-04-25 15:16:37

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import contextlib
import datetime
import logging
import os
import os.path as osp
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data_loader import load_dataset_train
from loss import CompareAlignLoss
from models.JIPNet import JIPNet


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(
    model,
    train_dataloader,
    valid_dataloader,
    device,
    cfg,
    save_dir=None,
    save_checkpoint=15,
    rank=0,
    train_sampler=None,
):
    lr = cfg["train_cfg"]["lr"]
    end_lr = cfg["train_cfg"]["end_lr"]
    optim_name = cfg["train_cfg"]["optimizer"]
    scheduler_type = cfg["train_cfg"]["scheduler_type"]
    num_epoch = cfg["train_cfg"]["epochs"]
    accum_steps = cfg["train_cfg"].get("accum_steps", 1)
    use_amp = cfg["train_cfg"].get("use_amp", True)

    valid = valid_dataloader is not None
    is_main = rank == 0

    criterion = CompareAlignLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if optim_name == "sgd":
        optimizer = torch.optim.SGD(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=0,
        )
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=1e-3,
        )
    elif optim_name == "adamW":
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=1e-2,
        )

    if scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=end_lr)
    elif scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, 15, 0.1)

    for epoch in range(num_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_losses = {"total": 0.0, "focal": 0.0, "Lr": 0.0}

        if is_main:
            logging.info(
                "epoch: {}, lr:{:.8f}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            pbar = tqdm(train_dataloader, desc=f"epoch:{epoch}, train")
        else:
            pbar = train_dataloader

        optimizer.zero_grad()

        for step, (input1, input2, align_target, target) in enumerate(pbar):
            input1 = input1.float().to(device)
            input2 = input2.float().to(device)
            align_target = align_target.float().to(device)
            target = target[:, 0:1].float().to(device)

            is_last_step = (step + 1) == len(train_dataloader)
            is_accum_step = ((step + 1) % accum_steps == 0) or is_last_step

            # Skip DDP gradient sync on intermediate accumulation steps
            sync_ctx = contextlib.nullcontext() if is_accum_step else model.no_sync()

            with sync_ctx:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    cla_pred, align_pred = model([input1, input2])

                # Loss always in FP32 — prevents NaN from FP16 log/pow overflow
                # (FP16 epsilon=1e-9 rounds to zero, causing log(0)=-inf → NaN)
                loss, items = criterion(
                    cla_pred.float(), target, align_pred.float(), align_target
                )
                scaled_loss = loss / accum_steps
                scaler.scale(scaled_loss).backward()

            for k in items:
                train_losses[k] += items[k] / len(train_dataloader)
            train_losses["total"] += loss.item() / len(train_dataloader)

            if is_main:
                pbar.set_postfix(loss=loss.item())

            if is_accum_step:
                # Unscale first so clip_grad_norm sees real gradient magnitudes
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            del loss, scaled_loss

        if is_main:
            pbar.close()
            logging_info = "\tTRAIN epoch {}: ".format(epoch)
            for k in train_losses:
                logging_info += "{}:{:.4f}, ".format(k, train_losses[k])
            logging.info(logging_info)

        scheduler.step()

        if is_main and save_dir is not None and epoch > save_checkpoint:
            state = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )
            torch.save(state, osp.join(save_dir, f"epoch_{epoch}.pth"))

        if not valid:
            continue

        # Use model.module directly to bypass DDP collective ops.
        # If we call model.forward() (the DDP wrapper) only on rank 0,
        # the DDP _sync_params BROADCAST waits for rank 1 forever → NCCL timeout.
        val_model = model.module if hasattr(model, "module") else model
        val_model.eval()
        with torch.no_grad():
            valid_losses = {"total": 0.0, "focal": 0.0, "Lr": 0.0}
            pbar = (
                tqdm(valid_dataloader, desc=f"epoch:{epoch}, val")
                if is_main
                else valid_dataloader
            )

            for input1, input2, align_target, target in pbar:
                input1 = input1.float().to(device)
                input2 = input2.float().to(device)
                align_target = align_target.float().to(device)
                target = target[:, 0:1].float().to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    cla_pred, align_pred = val_model([input1, input2])
                loss, items = criterion(
                    cla_pred.float(), target, align_pred.float(), align_target
                )

                for k in items:
                    valid_losses[k] += items[k] / len(valid_dataloader)
                valid_losses["total"] += loss.item() / len(valid_dataloader)
                del loss

            if is_main:
                pbar.close()
                logging_info = "\tVALID epoch {}: ".format(epoch)
                for k in valid_losses:
                    logging_info += "{}:{:.4f}, ".format(k, valid_losses[k])
                logging.info(logging_info)

        # Restore DDP model to train mode for next epoch
        model.train()


def main_worker(rank, world_size, gpu_ids, cfg, save_dir, train_info, valid_info):
    set_seed(seed=7 + rank)

    # logging.basicConfig must be called inside the child process spawned by
    # mp.spawn — the parent's file handler is NOT inherited across process
    # boundaries, so the log file would be empty if configured in __main__.
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
            filename=osp.join(save_dir, "info.log"),
            filemode="w",
        )

    local_gpu = gpu_ids[rank]
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_gpu)
    device = torch.device(f"cuda:{local_gpu}")

    model = JIPNet(
        input_size=cfg["model_cfg"]["input_size"],
        img_channel=cfg["model_cfg"]["img_channel"],
        num_classes=cfg["model_cfg"]["num_classes"],
        width=cfg["model_cfg"]["width"],
        enc_blk_nums=cfg["model_cfg"]["enc_blk_nums"],
        dw_expand=cfg["model_cfg"]["dw_expand"],
        ffn_expand=cfg["model_cfg"]["ffn_expand"],
        mid_blk_nums=cfg["model_cfg"]["mid_blk_nums"],
        mid_blk_strides=cfg["model_cfg"]["mid_blk_strides"],
        mid_embed_dims=cfg["model_cfg"]["mid_embed_dims"],
        dec_hidden_dim=cfg["model_cfg"]["dec_hidden_dim"],
        dec_nhead=cfg["model_cfg"]["dec_nhead"],
        dec_local_num=cfg["model_cfg"]["dec_local_num"],
        encoder_pretrain_pth=cfg["pretrain_cfg"]["encoder_pth"],
    ).to(device)

    model = DDP(model, device_ids=[local_gpu], output_device=local_gpu)

    # Total batch_size is split evenly across GPUs
    batch_per_gpu = max(1, cfg["train_cfg"]["batch_size"] // world_size)
    accum_steps = cfg["train_cfg"].get("accum_steps", 1)
    use_amp = cfg["train_cfg"].get("use_amp", True)

    train_dataset = load_dataset_train(
        info_lst=train_info["info_lst"],
        patch_size=cfg["model_cfg"]["input_size"],
        use_augmentation=True,
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_per_gpu,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = None
    if rank == 0 and valid_info is not None:
        valid_dataset = load_dataset_train(
            info_lst=valid_info["info_lst"],
            patch_size=cfg["model_cfg"]["input_size"],
            use_augmentation=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_per_gpu,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    if rank == 0:
        eff_batch = batch_per_gpu * world_size * accum_steps
        logging.info("Model          : {}".format(cfg["model_cfg"]["model"]))
        logging.info("World size     : {}  (GPUs: {})".format(world_size, gpu_ids))
        logging.info("Batch/GPU      : {}".format(batch_per_gpu))
        logging.info("Accum steps    : {}".format(accum_steps))
        logging.info("Effective batch: {}".format(eff_batch))
        logging.info("AMP enabled    : {}".format(use_amp))
        logging.info("Train samples  : {}".format(len(train_dataset)))
        logging.info("******** begin training ********")

    train(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        device=device,
        cfg=cfg,
        save_dir=save_dir if rank == 0 else None,
        save_checkpoint=cfg["train_cfg"]["epochs"] - 6,
        rank=rank,
        train_sampler=train_sampler,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    current_path = os.path.abspath(__file__)
    config_path = osp.join(osp.dirname(current_path), "configs", "JIPNet.yaml")
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)

    # set save dir
    save_basedir = osp.join(cfg["save_cfg"]["save_basedir"], cfg["model_cfg"]["model"])
    if cfg["save_cfg"]["save_title"] == "time":
        now = datetime.datetime.now()
        save_dir = osp.join(save_basedir, now.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        save_dir = osp.join(save_basedir, cfg["save_cfg"]["save_title"])

    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(save_dir, "config.yaml"))

    train_info = np.load(cfg["db_cfg"]["train_info_path"], allow_pickle=True).item()
    valid_info = np.load(cfg["db_cfg"]["valid_info_path"], allow_pickle=True).item()

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    gpu_ids = cfg["train_cfg"]["cuda_ids"]
    world_size = len(gpu_ids)

    mp.spawn(
        main_worker,
        args=(world_size, gpu_ids, cfg, save_dir, train_info, valid_info),
        nprocs=world_size,
        join=True,
    )
