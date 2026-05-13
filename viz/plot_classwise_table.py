import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
# Vehicle = car IoU × 100 (most representative single vehicle class)
# Cyclist  = avg(bicycle, motorcycle) × 100
# Pedestrian = pedestrian × 100
# Driv. Area = driveable_surface × 100  (— pending re-eval)
# mIoU = overall mIoU × 100
# Sources: Swin-B / ResNet-101 / ConvNeXt-B from full 16k-image val inference
#          SegFormer from 5% val split (currently retraining)

rows = [
    # backbone,          Vehicle,  Cyclist,  Pedestrian,  mIoU
    # Vehicle = car IoU  Cyclist = avg(bicycle, motorcycle)
    ("Swin-B + UPerNet",      65.3,   30.8,     36.5,       35.4),
    ("ResNet-101 + UPerNet*", 69.6,   30.0,     34.3,       35.6),
    ("ConvNeXt-B + UPerNet",  59.4,   11.8,     26.0,       24.4),
    ("SegFormer + UPerNet†",  66.0,   20.0,     25.3,       29.8),
]

cols = ["Image Backbone", "Vehicle", "Cyclist", "Pedestrian", "mIoU"]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis("off")
fig.patch.set_facecolor("white")

# Build cell data
cell_data = [[r[0]] + [f"{v:.1f}" if isinstance(v, float) else v for v in r[1:]] for r in rows]

table = ax.table(
    cellText=cell_data,
    colLabels=cols,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.2)

# ── Styling ───────────────────────────────────────────────────────────────────
header_color  = "#2C3E50"
row_colors    = ["#FAFAFA", "#F0F0F0"]
best_color    = "#EAF4EA"   # light green highlight for Swin-B mIoU col
mIoU_col      = len(cols) - 1

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    cell.set_linewidth(0.8)

    if row == 0:
        # Header
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=12)
    else:
        # Alternate row shading
        cell.set_facecolor(row_colors[(row - 1) % 2])
        cell.set_text_props(fontsize=12)

        # Left-align backbone column
        if col == 0:
            cell.set_text_props(ha="left", fontsize=12)
            cell._loc = "left"

        # Green highlight best mIoU — Swin-B is best non-preliminary model
        if row == 1 and col == mIoU_col:
            cell.set_facecolor(best_color)
            cell.set_text_props(fontweight="bold", color="#1A6B1A", fontsize=12)

        # Bold best per-column (find best numeric value per column)
        col_vals = [r[col] for r in rows if isinstance(r[col], float)]
        if isinstance(rows[row - 1][col], float) and col in (1, 2, 3):
            if rows[row - 1][col] == max(col_vals):
                cell.set_text_props(fontweight="bold", fontsize=12)

# Title
ax.set_title("TABLE I\nClasswise mIoU Comparison on the NuImages Dataset",
             fontsize=14, fontweight="bold", pad=18, loc="left", color="#2C3E50")

# Footnotes
fig.text(0.01, 0.02,
         "* ResNet-101 result is preliminary — trained on 5% of the dataset.\n"
         "† SegFormer trained on 5% split; per-class IoU evaluated on 5% val split.\n"
         "  Vehicle = car IoU. Cyclist = avg(bicycle, motorcycle). All values × 100.",
         fontsize=9, color="#666666", style="italic", va="bottom")

plt.tight_layout(rect=[0, 0.14, 1, 1])
out = "outputs/classwise_table.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")