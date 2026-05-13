import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Swin-B, ResNet101, ConvNeXt-B: full 16k-image val inference
# SegFormer: 5pct val mIoU (full dataset result is not representative)
# DINOv2: external reference

models  = ["DINO\n(Reference)", "Swin-B\n+ UPerNet", "ResNet-101\n+ UPerNet*", "ConvNeXt-B\n+ UPerNet", "SegFormer\n+ UPerNet"]
miou    = [0.4128,                 0.3539,               0.3559,                   0.2438,                  0.2978]
colors  = ["#4C72B0",              "#55A868",             "#C44E52",                "#CCB974",               "#8172B2"]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor("#F8F8F8")
ax.set_facecolor("#F8F8F8")

x    = np.arange(len(models))
bars = ax.bar(x, miou, color=colors, width=0.55, edgecolor="white", linewidth=1.5)

# Fade reference + preliminary bars
bars[0].set_alpha(0.70)  # DINOv2
bars[2].set_alpha(0.60)  # ResNet-101

# Value labels
for bar, val in zip(bars, miou):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=13, fontweight="bold", color="#333333")

# Best model annotation
ax.text(x[1], miou[1] + 0.024, "★ Best",
        ha="center", va="bottom", fontsize=12, color="#B8860B", fontweight="bold")

# DINOv2 reference line
ax.axhline(y=0.4128, color="#4C72B0", linestyle="--", linewidth=1.2, alpha=0.5)
ax.text(len(models) - 0.48, 0.4128 + 0.005, "DINO reference",
        ha="right", fontsize=11, color="#4C72B0", alpha=0.8)

# Axes
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=14, fontweight="bold")
ax.set_ylabel("mIoU", fontsize=14, fontweight="bold")
ax.set_title("Semantic Segmentation — mIoU Comparison\nNuImages Autonomous Driving Dataset",
             fontsize=16, fontweight="bold", pad=16)
ax.set_ylim(0, 0.50)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.text(0.02, 0.01, "* ResNet-101 result is preliminary — trained on 5% of the dataset.",
         ha="left", fontsize=10, color="#666666", style="italic")

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = "outputs/miou_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
