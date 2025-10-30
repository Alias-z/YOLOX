#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Model architecture - Optimized for YOLOX-Nano
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (480, 480)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (480, 480)
        self.mosaic_prob = 0.5
        self.enable_mixup = False  # Disabled for small Nano model
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Dataset - Single food class
        self.num_classes = 1
        self.data_dir = "data/yolo"  # Path relative to project root
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # Training settings - Optimized for Nano model
        self.max_epoch = 100  # 100 epochs sufficient for Nano
        self.data_num_workers = 4
        self.eval_interval = 5  # Evaluate every 5 epochs
        self.print_interval = 10

        # Optimizer settings
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        # Augmentation - Weaker aug for small Nano model
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
