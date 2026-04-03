import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

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


class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels-first tensors [B, C, H, W].
    ConvNeXt uses this instead of BatchNorm.
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        # Normalize over channel dim (dim=1) for BCHW tensors
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# -------------------------
# ConvNeXt Block
# -------------------------

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block (Liu et al., 2022):
        depthwise 7x7 conv -> LayerNorm -> pointwise 1x1 (expand 4x) -> GELU
        -> pointwise 1x1 (contract) -> residual (with optional stochastic depth)

    Inverted bottleneck: expand channels by 4x in the middle.
    Uses depthwise conv (groups=dim) to mix spatial info efficiently.
    """
    def __init__(self, dim, drop_path_prob=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # Layer scale: learnable per-channel scalar, initialized small
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim, 1, 1))

        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return shortcut + self.drop_path(x)


# -------------------------
# ConvNeXt-B Encoder
# -------------------------

class ConvNeXtBEncoder(nn.Module):
    """
    ConvNeXt-B encoder backbone.
    Outputs 4 feature maps at strides 4, 8, 16, 32.
    out_channels = [128, 256, 512, 1024]

    Architecture (ConvNeXt-B config):
        Stem      : 4x4 conv, stride 4          -> 128ch, H/4
        Stage 0   : 3  ConvNeXtBlocks, 128ch
        Downsample: 2x2 conv stride 2           -> 256ch, H/8
        Stage 1   : 3  ConvNeXtBlocks, 256ch
        Downsample: 2x2 conv stride 2           -> 512ch, H/16
        Stage 2   : 27 ConvNeXtBlocks, 512ch
        Downsample: 2x2 conv stride 2           -> 1024ch, H/32
        Stage 3   : 3  ConvNeXtBlocks, 1024ch

    Reference: A ConvNet for the 2020s (Liu et al., 2022)
    Compared directly vs Swin-B using UPerNet on ADE20K in the paper.
    """
    def __init__(self, drop_path_rate=0.4, layer_scale_init=1e-6):
        super().__init__()

        dims   = [128, 256, 512, 1024]
        depths = [3, 3, 27, 3]
        self.out_channels = dims

        # Stochastic depth decay
        total_blocks = sum(depths)
        dpr          = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cur          = 0

        # Stem: aggressive 4x4 downsampling (equivalent to ViT patch embed)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # Stages + downsampling layers between stages
        self.stages      = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[i],
                              drop_path_prob=dpr[cur + j],
                              layer_scale_init=layer_scale_init)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

            # Downsample between stages (not after last)
            if i < 3:
                self.downsamplers.append(nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                ))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward(self, x):
        x  = self.stem(x)
        
        # Stage 0
        c1 = self.stages[0](x)

        # Stage 1
        x  = self.downsamplers[0](c1)
        c2 = self.stages[1](x)

        # Stage 2 — 27 blocks, use gradient checkpointing
        x  = self.downsamplers[1](c2)
        if self.training:
            for blk in self.stages[2]:
                x = ckpt(blk, x, use_reentrant=False)
            c3 = x
        else:
            c3 = self.stages[2](x)

        # Stage 3
        x  = self.downsamplers[2](c3)
        c4 = self.stages[3](x)

        return [c1, c2, c3, c4]


# -------------------------
# UPerNet Decoder
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

class ConvNeXtBUPerNet(nn.Module):
    """
    ConvNeXt-B encoder + UPerNet decoder for semantic segmentation.

    Args:
        num_classes (int): number of segmentation classes
        fpn_dim (int): UPerNet internal feature dimension
        dropout (float): dropout rate
    """
    def __init__(self, num_classes=12, fpn_dim=256, dropout=0.1):
        super().__init__()
        self.encoder     = ConvNeXtBEncoder()
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
