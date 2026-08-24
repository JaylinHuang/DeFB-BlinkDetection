"""Quick feasibility verification for Lite Track Model.

Tests:
  1. Model instantiation + parameter count
  2. Forward pass shape validation
  3. RoIAlign bridge → Stage 2 input shapes
  4. Dummy time-mamba integration
  5. Inference speed benchmark (vs original if available)
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# Add parent project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_model import build_lite_model, count_parameters, count_all_parameters
from stage1_decoder import box_cxcywh_to_xyxy


# ============================================================
# Dummy Time-Mamba (simulating team lead's module)
# ============================================================
class DummyTimeMamba(nn.Module):
    """Placeholder for the team lead's time-mamba blink detector.

    Accepts the standard DeFB Stage 2 input interface:
      - blink_features: [B, T, 15360]  RoI-aligned eye features
      - head_query:     [B, T, 256]    head branch query features
      - eye_query:      [B, T, 256]    eye branch query features

    Returns: [B, T, 2] blink/not-blink logits
    """
    def __init__(self, feature_dim=15360, query_dim=256, hidden_dim=256):
        super().__init__()
        # Project the large RoI features down
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.mamba = nn.Identity()  # placeholder for actual mamba block
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, blink_features, head_query, eye_query):
        # blink_features: [B, T, 15360]
        # head_query:     [B, T, 256]
        # eye_query:      [B, T, 256]
        x = self.feat_proj(blink_features)  # [B, T, hidden_dim]
        x = x + head_query + eye_query      # fuse queries
        x = self.mamba(x)                   # temporal modeling
        return self.head(x)                 # [B, T, 2]


# ============================================================
# Bridge function (replicates test.py get_output logic)
# ============================================================
def bridge_to_stage2(output, memory, head_query, eye_query,
                     person_threshold=0.5, sample_point=(5, 4)):
    """Simulate the Stage1→Stage2 bridge from test.py get_output().

    This is the exact logic that extracts RoIAlign features and prepares
    them for the blink detection module (time-mamba).
    """
    T = head_query.size(0)  # head_query: [T, N, D] or [T, N, D]

    head_bbox = output['pred_head_boxes']  # [T, N, 4]  (N=50 for lite)
    eye_bbox = output['pred_eye_boxes']    # [T, N, 4]
    out_logits = output['pred_logits']     # [T, N, 1]

    N = out_logits.size(1)

    # Filter persons by confidence (top-10 frame average > threshold)
    out_logits = out_logits.reshape(T, N)
    out_logits = F.sigmoid(out_logits)
    out_logits = out_logits.transpose(0, 1)  # [N, T]
    score, _ = torch.topk(out_logits, k=min(10, T), dim=-1)
    cls_score = torch.mean(score, dim=-1)    # [N]
    save_index = cls_score > person_threshold

    if save_index.sum() == 0:
        print("  [WARNING] No persons detected above threshold, using all queries")
        save_index = torch.ones(N, dtype=torch.bool)

    K = save_index.sum().item()
    print(f"  Filtered: {K} persons (from {N} queries)")

    # Filter
    head_bbox = head_bbox[:, save_index, :]    # [T, K, 4]
    eye_bbox = eye_bbox[:, save_index, :]      # [T, K, 4]
    head_query = head_query[:, save_index, :]  # [T, K, 256]
    eye_query = eye_query[:, save_index, :]    # [T, K, 256]

    # RoIAlign on eye bboxes (3 scales × 5×4 = 60 points × 256 = 15360)
    device = eye_bbox.device
    T_K = T * K
    eye_roi = box_cxcywh_to_xyxy(eye_bbox.reshape(T_K, -1))

    det_blinks_eye = []
    for i, mem in enumerate(memory[-3:]):  # last 3 feature map scales
        _, C, H, W = mem.size()
        whwh = torch.tensor([[W, H, W, H]], dtype=torch.float32, device=device)
        scale_roi = eye_roi * whwh

        # roi_align expects boxes as list of [K, 4] per batch element
        boxes_list = [scale_roi[j:j+K] for j in range(0, T_K, K)]
        feature_map = roi_align(mem, boxes_list, sample_point)  # [T*K, C, 5, 4]
        feature_map = feature_map.permute(0, 2, 3, 1)          # [T*K, 5, 4, C]
        feature_map = feature_map.reshape(T, K, -1, C)          # [T, K, 20, C]
        det_blinks_eye.append(feature_map)

    det_blinks_eye = torch.cat(det_blinks_eye, dim=2)  # [T, K, 60, 256]
    det_blinks_eye = det_blinks_eye.reshape(T, K, -1)  # [T, K, 15360]
    det_blinks_eye = det_blinks_eye.transpose(0, 1)    # [K, T, 15360]

    head_query = head_query.permute(1, 0, 2)   # [K, T, 256]
    eye_query = eye_query.permute(1, 0, 2)     # [K, T, 256]

    # Windowing (simulating sliding window for longer sequences)
    window_size = 16
    stride = 8

    if T < window_size:
        # Pad
        pad_len = window_size - T
        blink_features = F.pad(det_blinks_eye, (0, 0, 0, pad_len))  # [K, 16, 15360]
        head_q = F.pad(head_query, (0, 0, 0, pad_len))              # [K, 16, 256]
        eye_q = F.pad(eye_query, (0, 0, 0, pad_len))                # [K, 16, 256]
        num_windows = 1
    else:
        num_windows = (T - window_size) // stride + 1
        windows_feat, windows_head, windows_eye = [], [], []
        for w in range(num_windows):
            start = w * stride
            end = start + window_size
            windows_feat.append(det_blinks_eye[:, start:end, :])
            windows_head.append(head_query[:, start:end, :])
            windows_eye.append(eye_query[:, start:end, :])
        blink_features = torch.cat(windows_feat, dim=0)   # [K*W, 16, 15360]
        head_q = torch.cat(windows_head, dim=0)           # [K*W, 16, 256]
        eye_q = torch.cat(windows_eye, dim=0)             # [K*W, 16, 256]

    return blink_features, head_q, eye_q, K, num_windows


# ============================================================
# Tests
# ============================================================

def test_all(device='cpu', pretrained=True):
    """Run all feasibility tests."""
    results = {'passed': 0, 'failed': 0, 'tests': []}

    def check(condition, name, detail=""):
        if condition:
            results['passed'] += 1
            results['tests'].append(f"  [PASS] {name}")
        else:
            results['failed'] += 1
            results['tests'].append(f"  [FAIL] {name}: {detail}")

    print("=" * 60)
    print("DeFB Lite Track Model -- Feasibility Verification")
    print("=" * 60)

    # ---- Test 1: Model Instantiation & Parameter Count ----
    print("\n[Test 1] Model Instantiation & Parameter Count")
    pretrained_path = None
    if pretrained:
        # Check common locations
        candidates = [
            "ResNet18_vd_pretrained_from_paddle.pth",
            os.path.expanduser("~/.cache/torch/hub/checkpoints/ResNet18_vd_pretrained_from_paddle.pth"),
            "E:/documents/mssb code/DeFB-main/ResNet18_vd_pretrained_from_paddle.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                pretrained_path = c
                break

    try:
        model = build_lite_model(
            backbone_pretrained=pretrained_path if pretrained_path else False,
            device=device,
        )
        total_params = count_all_parameters(model)
        trainable_params = count_parameters(model)

        print(f"  Total parameters:     {total_params:>10,}")
        print(f"  Trainable parameters: {trainable_params:>10,}")

        # Check parameter budget
        check(total_params < 30_000_000,
              f"Total params {total_params/1e6:.1f}M < 30M target",
              f"Got {total_params/1e6:.1f}M")
        check(total_params < 25_000_000,
              f"Total params {total_params/1e6:.1f}M < 25M stretch goal")

        # Check trainable ratio (backbone+encoder frozen ≈ 35% trainable)
        trainable_pct = trainable_params / total_params * 100
        print(f"  Trainable ratio: {trainable_pct:.1f}%")

    except Exception as e:
        print(f"  [FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        results['failed'] += 1
        return results

    # ---- Test 2: Forward Pass Shape Validation ----
    print("\n[Test 2] Forward Pass Shape Validation")
    try:
        model.eval()
        T = 30
        x = torch.randn(1, T, 3, 720, 1280).to(device)

        t0 = time.time()
        with torch.no_grad():
            out, memory, head_query, eye_query = model(x, test=True)
        t_forward = time.time() - t0

        print(f"  Input:  {list(x.shape)}")
        print(f"  Forward time: {t_forward*1000:.1f} ms ({t_forward/T*1000:.1f} ms/frame)")

        # Check output shapes
        N_exp = 50  # lite num_queries
        check(out['pred_logits'].shape == (T, N_exp, 1),
              f"pred_logits shape {list(out['pred_logits'].shape)}",
              f"Expected ({T}, {N_exp}, 1)")
        check(out['pred_head_boxes'].shape == (T, N_exp, 4),
              f"pred_head_boxes shape {list(out['pred_head_boxes'].shape)}",
              f"Expected ({T}, {N_exp}, 4)")
        check(out['pred_eye_boxes'].shape == (T, N_exp, 4),
              f"pred_eye_boxes shape {list(out['pred_eye_boxes'].shape)}",
              f"Expected ({T}, {N_exp}, 4)")
        check(head_query.shape == (T, N_exp, 256),
              f"head_query shape {list(head_query.shape)}",
              f"Expected ({T}, {N_exp}, 256)")
        check(eye_query.shape == (T, N_exp, 256),
              f"eye_query shape {list(eye_query.shape)}",
              f"Expected ({T}, {N_exp}, 256)")
        check(len(memory) == 3,
              f"memory levels: {len(memory)}",
              f"Expected 3")

        for i, mem in enumerate(memory):
            exp_h = 720 // (8 * (2**i))
            exp_w = 1280 // (8 * (2**i))
            check(mem.shape[0] == T and mem.shape[1] == 256,
                  f"memory[{i}] shape {list(mem.shape)} (ch=256 OK)",
                  f"Expected [{T}, 256, ~{exp_h}, ~{exp_w}]")

    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        results['failed'] += 1
        return results

    # ---- Test 3: RoIAlign Bridge → Stage 2 Shapes ----
    print("\n[Test 3] RoIAlign Bridge → Stage 2 Input Shapes")
    try:
        blink_feat, head_q, eye_q, K, W = bridge_to_stage2(
            out, memory, head_query, eye_query)

        print(f"  Persons detected: K={K}, Windows: W={W}")
        print(f"  blink_features: {list(blink_feat.shape)}")
        print(f"  head_query:     {list(head_q.shape)}")
        print(f"  eye_query:      {list(eye_q.shape)}")

        B_stage2 = K * W
        check(blink_feat.shape == (B_stage2, 16, 15360),
              f"blink_features shape {list(blink_feat.shape)}",
              f"Expected ({B_stage2}, 16, 15360)")
        check(head_q.shape == (B_stage2, 16, 256),
              f"head_query shape {list(head_q.shape)}",
              f"Expected ({B_stage2}, 16, 256)")
        check(eye_q.shape == (B_stage2, 16, 256),
              f"eye_query shape {list(eye_q.shape)}",
              f"Expected ({B_stage2}, 16, 256)")

    except Exception as e:
        print(f"  [FAIL] Bridge failed: {e}")
        import traceback
        traceback.print_exc()
        results['failed'] += 1
        return results

    # ---- Test 4: Dummy Time-Mamba Integration ----
    print("\n[Test 4] Dummy Time-Mamba Integration")
    try:
        mamba = DummyTimeMamba().to(device)

        with torch.no_grad():
            pred = mamba(blink_feat.to(device), head_q.to(device), eye_q.to(device))

        print(f"  Input shapes:  {list(blink_feat.shape)}, {list(head_q.shape)}, {list(eye_q.shape)}")
        print(f"  Output shape:  {list(pred.shape)}")

        check(pred.shape == (B_stage2, 16, 2),
              f"time-mamba output shape {list(pred.shape)}",
              f"Expected ({B_stage2}, 16, 2)")

        print("  [OK] End-to-end integration successful!")

    except Exception as e:
        print(f"  [FAIL] Integration failed: {e}")
        import traceback
        traceback.print_exc()
        results['failed'] += 1
        return results

    # ---- Test 5: Inference Speed Benchmark ----
    print("\n[Test 5] Inference Speed Benchmark")
    try:
        model.eval()
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model(torch.randn(1, 10, 3, 720, 1280).to(device), test=True)

        # Benchmark
        times = []
        for T in [10, 30, 42]:
            x = torch.randn(1, T, 3, 720, 1280).to(device)
            t0 = time.time()
            for _ in range(10):
                with torch.no_grad():
                    model(x, test=True)
            t_avg = (time.time() - t0) / 10
            ms_per_frame = t_avg / T * 1000
            times.append((T, t_avg, ms_per_frame))
            print(f"  T={T:>3}: {t_avg*1000:>6.1f}ms total, {ms_per_frame:>5.1f}ms/frame")

        # Check speed target (< 8 ms/frame on CPU is unrealistic, just report)
        print(f"  (Original DeFB: ~8.7ms/frame on GPU)")

    except Exception as e:
        print(f"  [WARN] Speed benchmark error (non-critical): {e}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")
    for t in results['tests']:
        print(t)
    print("=" * 60)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-pretrained', action='store_true', help='Skip pretrained backbone')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results = test_all(device=args.device, pretrained=not args.no_pretrained)

    if results['failed'] > 0:
        sys.exit(1)
    else:
        print("\n[SUCCESS] All tests passed! Lite model is ready for training.")
