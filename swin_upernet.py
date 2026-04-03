import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# -------------------------
# Helpers
# -------------------------

class DropPath(nn.Module):
    """Stochastic depth — drop entire residual path during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob     = 1 - self.drop_prob
        shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * random_tensor.floor()


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    Args:
        x: [B, H, W, C]
        window_size (int): window height = width
    Returns:
        windows: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """
    Reverse window_partition.
    Args:
        windows: [num_windows*B, window_size, window_size, C]
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# -------------------------
# Patch Embedding
# -------------------------

class PatchEmbed(nn.Module):
    """
    Split image into non-overlapping patches via Conv2d.
    patch_size=4, stride=4 → H/4 x W/4 tokens.
    """
    def __init__(self, in_chans=3, embed_dim=128, patch_size=4, layer_norm_eps=1e-5):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(self, x):
        x = self.proj(x)                           # [B, C, H/4, W/4]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # [B, N, C]
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """
    Downsample by 2x: concatenate 4 neighbouring patches → Linear projection.
    Halves H and W, doubles channels.
    """
    def __init__(self, in_dim, layer_norm_eps=1e-5):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * in_dim, eps=layer_norm_eps)
        self.reduction = nn.Linear(4 * in_dim, 2 * in_dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        pad_bottom = H % 2
        pad_right  = W % 2
        if pad_bottom or pad_right:
            x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
            H_pad = H + pad_bottom  # ← track padded size
            W_pad = W + pad_right
        else:
            H_pad, W_pad = H, W     # ← no change needed

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H_pad // 2, W_pad // 2  # ← return consistent dims


# -------------------------
# Window Attention
# -------------------------

class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self Attention (W-MSA / SW-MSA).
    Uses relative position bias.

    Args:
        dim (int): token channels
        window_size (int): window height = width
        num_heads (int)
        shift (bool): if True, apply cyclic shift (SW-MSA)
    """
    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0,
                 layer_norm_eps=1e-5):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # Relative position bias table: (2W-1) x (2W-1), one per head
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, W, W]
        coords_flat = coords.flatten(1)                                             # [2, W*W]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]                    # [2, N, N]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_pos_index = rel.sum(-1)
        self.register_buffer("relative_position_index", rel_pos_index)

        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: [B_windows, N_tokens, C]
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = attn + bias

        # Add shift mask for SW-MSA
        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


# -------------------------
# Swin Transformer Block
# -------------------------

class SwinBlock(nn.Module):
    """
    One Swin Transformer block: W-MSA or SW-MSA + MLP.
    shift=False → W-MSA (no shift)
    shift=True  → SW-MSA (cyclic shift by window_size//2)
    """
    def __init__(self, dim, num_heads, window_size=7, shift=False,
                 mlp_ratio=4.0, dropout=0.0, attn_drop=0.0,
                 drop_path_prob=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.window_size  = window_size
        self.shift_size   = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn  = WindowAttention(dim, window_size=window_size, num_heads=num_heads,
                                     attn_drop=attn_drop, proj_drop=dropout,
                                     layer_norm_eps=layer_norm_eps)
        self.norm2     = nn.LayerNorm(dim, eps=layer_norm_eps)
        hidden_dim     = int(dim * mlp_ratio)
        self.mlp       = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path_prob)

    def _get_attn_mask(self, H, W, device):
        """Compute SW-MSA attention mask for cyclic-shift."""
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to multiple of window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            x    = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self._get_attn_mask(Hp, Wp, x.device)
        else:
            mask = None

        # Partition into windows
        x_win = window_partition(x, self.window_size)             # [nW*B, ws, ws, C]
        x_win = x_win.view(-1, self.window_size ** 2, C)

        # Attention
        x_win = self.attn(x_win, mask=mask)
        x_win = x_win.view(-1, self.window_size, self.window_size, C)

        # Reverse windows
        x = window_reverse(x_win, self.window_size, Hp, Wp)       # [B, Hp, Wp, C]

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Remove padding
        x = x[:, :H, :W, :].contiguous().view(B, H * W, C)

        # Residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -------------------------
# Swin-B Encoder
# -------------------------

class SwinBEncoder(nn.Module):
    """
    Swin-B encoder backbone.
    Outputs 4 feature maps at strides 4, 8, 16, 32.
    out_channels = [128, 256, 512, 1024]

    Architecture (Swin-B config):
        Stage 0: 2  blocks, embed=128, heads=4,  window=7
        Stage 1: 2  blocks, embed=256, heads=8,  window=7  + PatchMerging
        Stage 2: 18 blocks, embed=512, heads=16, window=7  + PatchMerging
        Stage 3: 2  blocks, embed=1024,heads=32, window=7  + PatchMerging
    """
    def __init__(self, window_size=7, mlp_ratio=4.0,
                 dropout=0.0, attn_drop=0.0, drop_path_rate=0.3,
                 layer_norm_eps=1e-5):
        super().__init__()

        embed_dims  = [128, 256, 512, 1024]
        num_heads   = [4, 8, 16, 32]
        depths      = [2, 2, 18, 2]
        self.out_channels = embed_dims

        # Patch embedding: 4x4 non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dims[0],
                                      patch_size=4, layer_norm_eps=layer_norm_eps)

        # Stochastic depth decay rule
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cur = 0

        # Build 4 stages
        self.stages        = nn.ModuleList()
        self.patch_merging = nn.ModuleList()  # between stages 0-1, 1-2, 2-3
        self.stage_norms   = nn.ModuleList()

        for i in range(4):
            stage = nn.ModuleList([
                SwinBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    shift=(j % 2 == 1),       # alternate W-MSA and SW-MSA
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    drop_path_prob=dpr[cur + j],
                    layer_norm_eps=layer_norm_eps,
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            self.stage_norms.append(nn.LayerNorm(embed_dims[i], eps=layer_norm_eps))
            cur += depths[i]

            # PatchMerging after stages 0, 1, 2 (not after last stage)
            if i < 3:
                self.patch_merging.append(
                    PatchMerging(embed_dims[i], layer_norm_eps=layer_norm_eps))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward(self, x):
        # Patch embedding
        x, H, W = self.patch_embed(x)   # [B, H/4*W/4, 128]

        outs = []
        from torch.utils.checkpoint import checkpoint

        for i in range(4):
            for blk in self.stages[i]:
                if self.training:
                    x = checkpoint(blk, x, H, W, use_reentrant=False)
                else:
                    x = blk(x, H, W)
                    
            # Normalize and reshape to feature map
            x    = self.stage_norms[i](x)
            B, _, C = x.shape
            feat = x.transpose(1, 2).reshape(B, C, H, W)
            outs.append(feat)

            # PatchMerging to next stage (except after last)
            if i < 3:
                x, H, W = self.patch_merging[i](x, H, W)

        return outs  # [c1, c2, c3, c4]


# -------------------------
# UPerNet Decoder (same as resnet101_upernet.py)
# -------------------------

class PPM(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        hidden = in_channels // 4
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ) for size in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + hidden * len(pool_sizes), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w   = x.shape[2], x.shape[3]
        pooled = []
        for stage in self.stages:
            if x.device.type == "mps":
                out = stage(x.cpu()).to(x.device)
            else:
                out = stage(x)
            pooled.append(F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat([x] + pooled, dim=1))


class UPerNetHead(nn.Module):
    def __init__(self, in_channels, num_classes, fpn_dim=256, dropout=0.1):
        super().__init__()
        assert len(in_channels) == 4

        self.ppm = PPM(in_channels[3])

        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ) for c in in_channels
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ) for _ in in_channels
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        c4 = self.ppm(c4)

        p1 = self.laterals[0](c1)
        p2 = self.laterals[1](c2)
        p3 = self.laterals[2](c3)
        p4 = self.laterals[3](c4)

        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode="bilinear", align_corners=False)
        p2 = p2 + F.interpolate(p3, size=p2.shape[2:], mode="bilinear", align_corners=False)
        p1 = p1 + F.interpolate(p2, size=p1.shape[2:], mode="bilinear", align_corners=False)

        p1 = self.smooth[0](p1)
        p2 = self.smooth[1](p2)
        p3 = self.smooth[2](p3)
        p4 = self.smooth[3](p4)

        target = p1.shape[2:]
        p2 = F.interpolate(p2, size=target, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=target, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=target, mode="bilinear", align_corners=False)

        fused  = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))
        return self.classifier(fused)


# -------------------------
# Full Model
# -------------------------

class SwinBUPerNet(nn.Module):
    """
    Swin-B encoder + UPerNet decoder for semantic segmentation.

    Args:
        num_classes (int): number of segmentation classes
        fpn_dim (int): UPerNet internal feature dimension
        dropout (float): dropout rate
    """
    def __init__(self, num_classes=12, fpn_dim=256, dropout=0.1):
        super().__init__()
        self.encoder     = SwinBEncoder()
        self.decode_head = UPerNetHead(
            in_channels=self.encoder.out_channels,
            num_classes=num_classes,
            fpn_dim=fpn_dim,
            dropout=dropout,
        )

    def forward(self, x):
        feats  = self.encoder(x)
        logits = self.decode_head(feats)
        logits = F.interpolate(logits, size=x.shape[2:],
                               mode="bilinear", align_corners=False)
        return logits
