import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse Rohit's encoder exactly as-is
from segformer import SegformerEncoder


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

class SegformerUPerNet(nn.Module):
    """
    SegFormer MiT encoder (Rohit's implementation) + UPerNet decoder.

    This lets us fairly compare:
        SegformerClasswise  → MiT encoder + MLP decoder (Rohit's original)
        SegformerUPerNet    → MiT encoder + UPerNet decoder (this file)

    Uses B0 config by default to match Rohit's SegformerClasswise.
    Change embed_dims/depths for B2/B4 variants.

    Args:
        num_classes (int)
        embed_dims  (tuple): MiT stage channel dims, default B0=(32,64,160,256)
        num_heads   (tuple)
        depths      (tuple)
        sr_ratios   (tuple)
        fpn_dim     (int): UPerNet internal channels
        dropout     (float)
    """
    def __init__(
        self,
        num_classes=12,
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4.0, 4.0, 4.0, 4.0),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        dropout=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        fpn_dim=256,
        decoder_dropout=0.1,
    ):
        super().__init__()
        self.encoder = SegformerEncoder(
            in_chans=3,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            dropout=dropout,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
            layer_norm_eps=layer_norm_eps,
        )
        self.decode_head = UPerNetHead(
            in_channels=list(embed_dims),
            num_classes=num_classes,
            fpn_dim=fpn_dim,
            dropout=decoder_dropout,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,   0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        feats  = self.encoder(x)           # [c1, c2, c3, c4]
        logits = self.decode_head(feats)   # [B, num_classes, H/4, W/4]
        logits = F.interpolate(logits, size=x.shape[2:],
                               mode="bilinear", align_corners=False)
        return logits
