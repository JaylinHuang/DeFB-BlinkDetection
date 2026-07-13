"""Training entry point for Lite Track Model.

Single GPU, small batch size. Run directly (no torchrun needed):

    python newtest/train_lite.py -c newtest/config_lite.yml --device cuda:0

Builds the model manually (no @register dependency), injects it into
the YAML config, then hands off to the parent project's DetSolver.
"""

import sys
import os
import argparse
import torch

# Add parent project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import YAMLConfig
from src.solver import TASKS
from src.misc import dist_utils

from newtest.track_model_lite import build_lite_model, build_nano_model
import newtest.collate  # register LiteBatchImageCollate via @register()


def get_pretrained_path():
    """Find PResNet-18 pretrained weights."""
    candidates = [
        "ResNet18_vd_pretrained_from_paddle.pth",
        os.path.join(os.path.dirname(__file__), "..", "ResNet18_vd_pretrained_from_paddle.pth"),
        os.path.expanduser("~/.cache/torch/hub/checkpoints/ResNet18_vd_pretrained_from_paddle.pth"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return False


def main():
    parser = argparse.ArgumentParser(description='Train Lite Track Model (Single GPU)')
    parser.add_argument('-c', '--config',
                        default=os.path.join(os.path.dirname(__file__), 'config_lite.yml'),
                        help='Config file path')
    parser.add_argument('-t', '--tuning', default=None,
                        help='Tuning checkpoint path')
    parser.add_argument('-r', '--resume', default=None,
                        help='Resume checkpoint path')
    parser.add_argument('-d', '--device', default=None,
                        help='Device (cpu, cuda:0, etc.) — default: cuda if available')
    parser.add_argument('--pretrained_backbone', default=None,
                        help='Path to PResNet-18 pretrained weights')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='Run validation only')
    parser.add_argument('--nano', action='store_true', default=False,
                        help='Use Nano model (MobileNetV3-Small, ~2.6M params) instead of Lite')
    args = parser.parse_args()

    # ---- Setup distributed (falls back to single-process gracefully) ----
    # When NOT using torchrun: RANK/LOCAL_RANK are unset, init_process_group fails,
    # setup_distributed catches the exception and falls back to single-process mode.
    dist_utils.setup_distributed(seed=args.seed)

    # ---- Device ----
    if args.device:
        device = args.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Single process (no torchrun)")

    # ---- Load YAML config ----
    print(f"\nLoading config: {args.config}")
    cfg = YAMLConfig(args.config)

    # ---- Build Model (Lite or Nano) ----
    if args.nano:
        print("\nBuilding Nano Track Model (MobileNetV3-Small, ~2.6M)...")
        model = build_nano_model(
            backbone_pretrained=True,
            eval_spatial_size=(720, 1280),
        )
    else:
        print("\nBuilding Lite Track Model...")
        pretrained_path = args.pretrained_backbone or get_pretrained_path()
        if pretrained_path:
            print(f"  Pretrained backbone: {pretrained_path}")
        else:
            print("  WARNING: No pretrained backbone found, using random init")

        model = build_lite_model(
            backbone_pretrained=pretrained_path if pretrained_path else False,
        )

    # Print parameter stats
    from newtest.track_model_lite import count_parameters, count_all_parameters
    total = count_all_parameters(model)
    trainable = count_parameters(model)
    print(f"  Total parameters:     {total:>10,} ({total/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable:>10,} ({trainable/1e6:.1f}M)")
    print(f"  Trainable ratio:      {trainable/total*100:.1f}%")

    # ---- Inject model and device into config ----
    # YAMLConfig overrides the model property getter, so set _model directly
    cfg._model = model
    cfg._device = device

    # Ensure single GPU compatible settings
    if not hasattr(cfg, 'sync_bn') or cfg.sync_bn is None:
        cfg.sync_bn = False
    if not hasattr(cfg, 'find_unused_parameters') or cfg.find_unused_parameters is None:
        cfg.find_unused_parameters = False

    # ---- Tuning / Resume ----
    if args.tuning:
        cfg.tuning = args.tuning
    if args.resume:
        cfg.resume = args.resume

    # ---- Delegate to existing Solver ----
    print(f"\nTask: {cfg.task}")
    solver_class = TASKS[cfg.task]
    solver = solver_class(cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':
    main()
