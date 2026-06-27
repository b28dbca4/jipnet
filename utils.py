"""
Description:
Author: Xiongjun Guan
Date: 2025-04-25 15:18:43
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-04-25 15:18:47

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import numpy as np
import torch

from models.utils import AffinePatch  # noqa: F401 — canonical definition in models/utils.py


def show_pairs(img1, img2, mode="gray"):
    img1 = img1[:, :, None]
    img2 = img2[:, :, None]
    if mode == "rgb":
        if np.max(img1) <= 1:
            img0 = np.ones_like(img1)
        else:
            img0 = np.ones_like(img1) * 255.0
        img = np.concatenate((img0, img1, img2), axis=2)
    elif mode == "gray":
        img = (img1 * 1.0 + img2 * 1.0) / 2
    return img


def load_model(model, ckp_path):

    def remove_module_string(k):
        items = k.split(".")
        items = items[0:1] + items[2:]
        return ".".join(items)

    if isinstance(ckp_path, str):
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
        ckp_model_dict = ckp["model"]
    else:
        ckp_model_dict = ckp_path

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {remove_module_string(k): v for k, v in ckp_model_dict.items()}

    if hasattr(model, "module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)


def translate_network_output(pred, translation_const):
    angs = np.rad2deg(np.arctan2(pred[:, 1], pred[:, 0]))
    txs = pred[:, 2] * translation_const
    tys = pred[:, 3] * translation_const

    return angs, txs, tys
