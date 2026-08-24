"""Custom collate function for Lite model training.
Fixes key name mismatch between dataset (head_bbox/eye_bbox) and criterion (head_boxes/eye_boxes).
"""

import sys
import os
import torch

# Register with the parent framework
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.core.workspace import register, GLOBAL_CONFIG


def lite_batch_collate_fn(items):
    """Collate function that renames head_bbox→head_boxes, eye_bbox→eye_boxes.

    Mirrors the original BatchImageCollateFuncion but uses keys expected by criterion.
    """
    clip_num = len(items)
    sample_num = len(items[0])

    targets = []
    T, C, H, W = items[0][0]['image'].size()

    images = torch.zeros(clip_num, sample_num, T, C, H, W)

    for i in range(clip_num):
        for j in range(sample_num):
            item = items[i][j]
            images[i, j] = item['image']
            targets.append({
                'head_boxes': item['head_bbox'],   # renamed for criterion
                'eye_boxes': item['eye_bbox'],     # renamed for criterion
                'labels': torch.zeros(T),
                'blink_gt': item['blink_gt']
            })

    return images.reshape(clip_num * sample_num, T, C, H, W), targets


@register()
class LiteBatchImageCollate:
    """Wrapper compatible with @register / YAML config system.

    Usage in YAML:
        collate_fn:
          type: LiteBatchImageCollate
    """
    def __init__(self, scales=None, stop_epoch=None):
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000

    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return getattr(self, '_epoch', -1)

    def __call__(self, items):
        return lite_batch_collate_fn(items)
