#!/usr/bin/env python3
"""
Custom optimizer utilities for YOLOv8 training.
"""

import logging
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer

logger = logging.getLogger(__name__)


class CustomDetectionTrainer(DetectionTrainer):
    """Custom YOLO trainer with custom optimizer support."""

    def __init__(self, custom_optimizer_class=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_optimizer_class = custom_optimizer_class

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build optimizer with custom optimizer support."""
        if self.custom_optimizer_class is None:
            return super().build_optimizer(model, name, lr, momentum, decay, iterations)

        logger.info(f"Using custom optimizer: {self.custom_optimizer_class.__name__}")

        # Separate parameters by weight decay
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g[1].append(param)
                else:
                    g[0].append(param)

        # Create custom optimizer
        optimizer = self.custom_optimizer_class([
            {'params': g[0], 'weight_decay': decay},
            {'params': g[1], 'weight_decay': 0.0},
            {'params': g[2], 'weight_decay': 0.0}
        ])

        return optimizer
