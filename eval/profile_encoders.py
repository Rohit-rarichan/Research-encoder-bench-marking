#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Inference latency profiler for all encoder+UPerNet models.

Reports per-image latency (ms) broken down into three phases:
  1. reshape  – patch-embedding tokenisation + final bilinear upsample
  2. encoder  – core backbone computation (attn/conv, norms, etc.)
  3. upernet  – UPerNetHead forward pass

Usage (synthetic inputs — default, recommended for benchmarking):
    python profile_encoders.py

Usage (real dataset images):
    python profile_encoders.py --data_root /path/to/nuimages --version v1.0-train
    python profile_encoders.py --data_root /path/to/nuimages --num_images 100 --batch_size 1
"""

import argparse
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

from models.segformer_upernet import SegformerUPerNet
from models.convnext_upernet import ConvNeXtBUPerNet
from models.swin_upernet import SwinBUPerNet
from models.resnet101_upernet import ResNet101UPerNet
from data.nuimages_dataset import NUM_CLASSES
from models.segformer import OverlapPatchEmbed
from models.swin_upernet import PatchEmbed as SwinPatchEmbed, PatchMerging as SwinPatchMerging


# ── timing primitives ────────────────────────────────────────────────────────

class StageTimer:
    """Records wall-clock or CUDA-event time for a named stage."""

    def __init__(self, device):
        self.device = device
        self._buf = defaultdict(list)
        self._start_event = {}
        self._start_cpu = {}

    def start(self, name):
        if self.device.type == "cuda":
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._start_event[name] = e
        else:
            self._start_cpu[name] = time.perf_counter()

    def stop(self, name):
        if self.device.type == "cuda":
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            torch.cuda.synchronize()
            self._buf[name].append(self._start_event[name].elapsed_time(e))
        else:
            self._buf[name].append((time.perf_counter() - self._start_cpu[name]) * 1e3)

    def stats(self, name):
        arr = np.array(self._buf[name])
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "n": 0}
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "median": float(np.median(arr)),
            "n":      len(arr),
        }


# ── reshape-hook instrumentation ─────────────────────────────────────────────

def attach_reshape_hooks(model, timer, device):
    """
    Attaches forward hooks to measure reshape time:
      - OverlapPatchEmbed (SegFormer) or SwinPatchEmbed (Swin)
        → tokenisation: image [B,C,H,W] → tokens [B,N,C]

    For CNN backbones (ConvNeXt, ResNet) reshape_time is zero by definition.

    Returns a list of hook handles so they can be removed later.
    """
    handles = []

    # Reshape ops: patch tokenisation (SegFormer/Swin) and Swin patch merging
    RESHAPE_TYPES = (OverlapPatchEmbed, SwinPatchEmbed, SwinPatchMerging)

    for module in model.encoder.modules():
        if isinstance(module, RESHAPE_TYPES):
            def _pre(m, inp, _timer=timer, _dev=device):
                if getattr(_timer, "_measuring", False):
                    _timer.start("_patch_embed")

            def _post(m, inp, out, _timer=timer, _dev=device):
                if getattr(_timer, "_measuring", False):
                    _timer.stop("_patch_embed")

            handles.append(module.register_forward_pre_hook(_pre))
            handles.append(module.register_forward_hook(_post))

    return handles


# ── per-model profiling ───────────────────────────────────────────────────────

@torch.inference_mode()
def profile_model(name, model, loader, device, warmup=10, num_images=100):
    model.eval()
    timer = StageTimer(device)

    # patch-embed reshape hooks (zero overhead for CNN models)
    reshape_handles = attach_reshape_hooks(model, timer, device)

    images_seen = 0
    batch_count = 0
    timer._measuring = False  # gate reshape hooks during warmup

    for images, _ in loader:
        images = images.to(device, non_blocking=True) if images.device != device else images

        # ── warmup ──────────────────────────────────────────────────────────
        if batch_count < warmup:
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_count += 1
            continue

        if images_seen >= num_images:
            break

        # ── encoder timing (reshape hooks fire inside here) ──────────────────
        timer._measuring = True
        timer.start("encoder")
        feats = model.encoder(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timer.stop("encoder")
        timer._measuring = False

        # ── upernet timing ────────────────────────────────────────────────────
        timer.start("upernet")
        logits = model.decode_head(feats)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timer.stop("upernet")

        # ── final upsample (also a reshape) ──────────────────────────────────
        timer.start("_final_upsample")
        _ = F.interpolate(logits, size=images.shape[2:],
                          mode="bilinear", align_corners=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timer.stop("_final_upsample")

        images_seen += images.shape[0]
        batch_count += 1

    # remove hooks
    for h in reshape_handles:
        h.remove()

    # ── aggregate ─────────────────────────────────────────────────────────────
    enc   = timer.stats("encoder")
    uper  = timer.stats("upernet")

    # reshape = all patch-embed calls within encoder + final upsample
    patch_embed_times = np.array(timer._buf.get("_patch_embed", []))
    final_up_times    = np.array(timer._buf.get("_final_upsample", []))

    # pair up reshape contributions per forward pass (N patch-embeds per image batch)
    enc_n = enc["n"]
    if enc_n == 0:
        reshape_mean = reshape_std = reshape_median = 0.0
    else:
        # sum all patch-embed elapsed times and final-upsample per batch
        pe_per_batch = (
            patch_embed_times.sum() / enc_n
            if len(patch_embed_times) > 0 else 0.0
        )
        fu_per_batch = (
            final_up_times.mean()
            if len(final_up_times) > 0 else 0.0
        )
        # per-batch reshape totals (used for std estimate)
        n_pe_per_batch = len(patch_embed_times) // enc_n if enc_n > 0 else 0
        if n_pe_per_batch > 0 and len(patch_embed_times) > 0:
            pe_batched = patch_embed_times.reshape(enc_n, n_pe_per_batch).sum(axis=1)
        else:
            pe_batched = np.zeros(enc_n)

        reshape_totals = pe_batched + final_up_times
        reshape_mean   = float(reshape_totals.mean())
        reshape_std    = float(reshape_totals.std())
        reshape_median = float(np.median(reshape_totals))

    return {
        "reshape": {"mean": reshape_mean, "std": reshape_std,
                    "median": reshape_median, "n": enc_n},
        "encoder": enc,
        "upernet": uper,
        "total_mean": reshape_mean + enc["mean"] + uper["mean"],
    }


# ── print table ──────────────────────────────────────────────────────────────

def print_results(all_results):
    col_w = 14
    header_w = 18

    models = list(all_results.keys())
    stages = ["reshape", "encoder", "upernet"]

    sep = "─" * (header_w + col_w * len(models) + 3)

    print(f"\n{'Latency per image (ms)':^{header_w + col_w * len(models)}}")
    print(sep)
    # header
    row = f"{'Stage':<{header_w}}"
    for m in models:
        row += f"{m:^{col_w}}"
    print(row)
    print(sep)

    for stage in stages:
        row = f"  {stage:<{header_w - 2}}"
        for m in models:
            s = all_results[m][stage]
            row += f"  {s['mean']:>5.2f}±{s['std']:>4.2f}  "
        print(row)

    print(sep)
    row = f"  {'total (mean)':<{header_w - 2}}"
    for m in models:
        row += f"  {all_results[m]['total_mean']:>10.2f}  "
    print(row)
    print(sep)

    # fraction breakdown
    print(f"\n{'Fraction of total latency':^{header_w + col_w * len(models)}}")
    print(sep)
    row = f"{'Stage':<{header_w}}"
    for m in models:
        row += f"{m:^{col_w}}"
    print(row)
    print(sep)
    for stage in stages:
        row = f"  {stage:<{header_w - 2}}"
        for m in models:
            total = all_results[m]["total_mean"]
            frac  = (all_results[m][stage]["mean"] / total * 100) if total > 0 else 0
            row  += f"    {frac:>5.1f} %   "
        print(row)
    print(sep)
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile encoder inference latency")
    parser.add_argument("--data_root", default="",
                        help="Path to nuImages dataset root (optional; "
                             "uses synthetic inputs if omitted)")
    parser.add_argument("--version",   default="v1.0-train",
                        help="nuImages version string (default: v1.0-train)")
    parser.add_argument("--num_images", type=int, default=100,
                        help="Number of images to profile over (default: 100)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size (default: 1)")
    parser.add_argument("--img_size",   type=int, default=512,
                        help="Input image size (default: 512)")
    parser.add_argument("--warmup",     type=int, default=10,
                        help="Warmup batches before timing (default: 10)")
    parser.add_argument("--device",     default="",
                        help="Force device: 'cpu', 'cuda', 'mps'. "
                             "Auto-detected if empty.")
    args = parser.parse_args()

    # device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # build data source: real dataset or synthetic random tensors
    total_batches = args.warmup + args.num_images  # batch_size=1 → 1 image/batch
    if args.data_root:
        from nuimages_dataset import NuImagesDataset
        print(f"Loading dataset from {args.data_root} ({args.version}) …")
        ds = NuImagesDataset(
            args.data_root,
            split="val",
            img_size=args.img_size,
            version=args.version,
            use_internal_split=False,
        )
        needed = min(len(ds), total_batches * args.batch_size + 1)
        loader = DataLoader(
            Subset(ds, list(range(needed))),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Using {needed} real images (val split).\n")
    else:
        # synthetic: pre-generate fixed random tensors, loop over them
        print(f"Using synthetic random inputs "
              f"({args.img_size}×{args.img_size}, batch={args.batch_size}).\n")
        dummy = torch.randn(
            args.batch_size, 3, args.img_size, args.img_size, device=device
        )
        dummy_label = torch.zeros(
            args.batch_size, args.img_size, args.img_size,
            dtype=torch.long, device=device
        )
        loader = [(dummy, dummy_label)] * total_batches

    # model registry
    model_configs = {
        "segformer": SegformerUPerNet(num_classes=NUM_CLASSES),
        "resnet101": ResNet101UPerNet(num_classes=NUM_CLASSES),
        "swin_b":    SwinBUPerNet(num_classes=NUM_CLASSES),
        "convnext_b": ConvNeXtBUPerNet(num_classes=NUM_CLASSES),
    }

    all_results = {}

    for model_name, model in model_configs.items():
        model = model.to(device)
        param_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Profiling {model_name:<12} ({param_m:.1f} M params) …", flush=True)

        result = profile_model(
            model_name, model, loader, device,
            warmup=args.warmup,
            num_images=args.num_images,
        )
        all_results[model_name] = result

        enc  = result["encoder"]
        uper = result["upernet"]
        rshp = result["reshape"]
        print(f"  reshape  {rshp['mean']:>7.2f} ± {rshp['std']:>5.2f} ms "
              f"(median {rshp['median']:.2f} ms)")
        print(f"  encoder  {enc['mean']:>7.2f} ± {enc['std']:>5.2f} ms "
              f"(median {enc['median']:.2f} ms)")
        print(f"  upernet  {uper['mean']:>7.2f} ± {uper['std']:>5.2f} ms "
              f"(median {uper['median']:.2f} ms)")
        print(f"  total    {result['total_mean']:>7.2f} ms  "
              f"(n={enc['n']} batches)\n")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results(all_results)


if __name__ == "__main__":
    main()
