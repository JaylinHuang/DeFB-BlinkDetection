"""Visualize training metrics from saved loss data."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'lite_quick')

with open(os.path.join(output_dir, 'loss_detail.json')) as f:
    d = json.load(f)

steps = d['step']

# Create figure with 3 panels
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('DeFB Lite Track Model - Quick Training (5 videos, 1 epoch)', fontsize=14, fontweight='bold')

# Panel 1: Total loss
ax = axes[0]
ax.plot(steps, d['loss_current'], alpha=0.2, color='blue', linewidth=0.5, label='Per-batch')
# Smooth with moving average
window = 20
if len(d['loss_avg']) >= window:
    smoothed = np.convolve(d['loss_current'], np.ones(window)/window, mode='valid')
    ax.plot(steps[window-1:], smoothed, color='blue', linewidth=2, label=f'Smoothed (w={window})')
ax.set_ylabel('Total Loss', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title(f'Total Loss: {d["loss_current"][0]:.0f} -> {d["loss_current"][-1]:.0f} '
             f'(avg: {d["loss_avg"][-1]:.0f})', fontsize=11)

# Panel 2: BBox losses
ax = axes[1]
ax.plot(steps, d['bbox_head'], alpha=0.3, color='#2196F3', linewidth=0.5, label='Head BBox L1')
ax.plot(steps, d['bbox_eye'], alpha=0.3, color='#FF9800', linewidth=0.5, label='Eye BBox L1')
# Smoothed
if len(steps) >= window:
    s_head = np.convolve(d['bbox_head'], np.ones(window)/window, mode='valid')
    s_eye = np.convolve(d['bbox_eye'], np.ones(window)/window, mode='valid')
    ax.plot(steps[window-1:], s_head, color='#2196F3', linewidth=2)
    ax.plot(steps[window-1:], s_eye, color='#FF9800', linewidth=2)
ax.set_ylabel('L1 Loss', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title(f'BBox Regression: head={d["bbox_head"][-1]:.0f} (avg), eye={d["bbox_eye"][-1]:.0f} (avg)', fontsize=11)

# Panel 3: GIoU losses
ax = axes[2]
ax.plot(steps, d['giou_head'], alpha=0.3, color='#4CAF50', linewidth=0.5, label='Head GIoU')
ax.plot(steps, d['giou_eye'], alpha=0.3, color='#E91E63', linewidth=0.5, label='Eye GIoU')
if len(steps) >= window:
    s_h = np.convolve(d['giou_head'], np.ones(window)/window, mode='valid')
    s_e = np.convolve(d['giou_eye'], np.ones(window)/window, mode='valid')
    ax.plot(steps[window-1:], s_h, color='#4CAF50', linewidth=2)
    ax.plot(steps[window-1:], s_e, color='#E91E63', linewidth=2)
ax.set_xlabel('Training Iteration', fontsize=12)
ax.set_ylabel('GIoU Loss', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title(f'GIoU: head={d["giou_head"][-1]:.1f} (avg), eye={d["giou_eye"][-1]:.1f} (avg)', fontsize=11)

plt.tight_layout()
save_path = os.path.join(output_dir, 'training_curves.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved: {save_path}')
plt.close()
