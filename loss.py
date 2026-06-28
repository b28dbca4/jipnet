"""
Description:
Author: Xiongjun Guan
Date: 2024-01-19 22:33:29
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-10-30 17:48:25

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-6


class BinaryFocalLoss(nn.Module):
    """
    https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=0.2, epsilon=1.0e-9):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input

        zero_hot_key = 1 - multi_hot_key
        loss = (
            -self.alpha
            * multi_hot_key
            * torch.pow((1 - logits), self.gamma)
            * (logits + self.epsilon).log()
        )
        loss += (
            -(1 - self.alpha)
            * zero_hot_key
            * torch.pow(logits, self.gamma)
            * (1 - logits + self.epsilon).log()
        )
        return loss.mean()


class SegmentationLoss(nn.Module):
    """Lseg from paper Eq. 10: simplified pixel-wise binary focal loss for the
    segmentation head used during enhancement pre-training.

    For each pixel p with prediction prob and ground-truth label y ∈ {0,1}:
        Lseg = -mean[ α*(1-p)^γ*log(p+ε)*y  +  (1-α)*p^γ*log(1-p+ε)*(1-y) ]

    Parameters match the main BinaryFocalLoss defaults (γ=2.0, α=0.2).
    """

    def __init__(self, gamma=2.0, alpha=0.2, epsilon=1.0e-9):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 1, H, W) — sigmoid probabilities from segmentation head
            target: (B, 1, H, W) — binary mask ground-truth (0 or 1)
        Returns:
            scalar loss
        """
        p = pred
        y = target
        loss = (
            -self.alpha
            * y
            * torch.pow((1 - p), self.gamma)
            * (p + self.epsilon).log()
        )
        loss += (
            -(1 - self.alpha)
            * (1 - y)
            * torch.pow(p, self.gamma)
            * (1 - p + self.epsilon).log()
        )
        return loss.mean()


class CompareAlignLoss(nn.Module):
    def __init__(self, w=0.002):
        super().__init__()
        self.focal_loss = BinaryFocalLoss()
        self.w = w

    def forward(
        self,
        cla_pred,
        cla_gt,
        align_pred,
        align_gt,
        lambda_2=0.99,
    ):
        focal_loss = self.focal_loss(
            cla_pred,
            cla_gt,
        )

        pred_b1 = align_pred[:, 0]
        pred_b2 = align_pred[:, 1]
        # eps must be INSIDE sqrt so the gradient 1/(2√x) stays bounded.
        # eps outside only prevents forward division-by-zero but gradient of
        # sqrt(0) is still ∞, which causes NaN at initialisation when pred≈0.
        norm = torch.sqrt(torch.square(pred_b1) + torch.square(pred_b2) + eps)
        cosT = pred_b1 / norm
        sinT = pred_b2 / norm
        pred_norm = torch.cat(
            [
                cosT[:, None],
                sinT[:, None],
                align_pred[:, 2][:, None],
                align_pred[:, 3][:, None],
            ],
            dim=1,
        )

        l2 = torch.square(pred_norm - align_gt)
        l2 = lambda_2 * (l2[:, 0] + l2[:, 1]) + (1 - lambda_2) * (l2[:, 2] + l2[:, 3])

        Lr = torch.sum(l2 * cla_gt.reshape((-1,))) / (torch.sum(cla_gt) + eps)

        loss = focal_loss + self.w * Lr
        items = {
            "focal": focal_loss.item(),
            "Lr": Lr.item(),
        }

        return loss, items
