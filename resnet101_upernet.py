import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic Building Blocks
# -------------------------

class Bottleneck(nn.Module):
    """
    ResNet Bottleneck block: 1x1 -> 3x3 -> 1x1 conv with residual connection.
    expansion=4 means output channels = planes * 4
    """
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample  # projects residual if shapes differ

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


# -------------------------
# ResNet-101 Encoder
# -------------------------

class ResNet101Encoder(nn.Module):
    """
    ResNet-101 encoder backbone.
    Outputs 4 feature maps at strides 4, 8, 16, 32.
    out_channels = [256, 512, 1024, 2048]

    Architecture:
        stem  : 7x7 conv, BN, ReLU, maxpool  -> stride 4
        layer1: 3  Bottleneck blocks          -> stride 4,  256ch
        layer2: 4  Bottleneck blocks          -> stride 8,  512ch
        layer3: 23 Bottleneck blocks          -> stride 16, 1024ch
        layer4: 3  Bottleneck blocks          -> stride 32, 2048ch
    """
    def __init__(self):
        super().__init__()

        self.out_channels = [256, 512, 1024, 2048]

        # Stem
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(64,   64,  blocks=3,  stride=1)
        self.layer2 = self._make_layer(256,  128, blocks=4,  stride=2)
        self.layer3 = self._make_layer(512,  256, blocks=23, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3,  stride=2)

        self._init_weights()

    def _make_layer(self, in_channels, planes, blocks, stride):
        """Build one residual stage."""
        downsample = None
        out_channels = planes * Bottleneck.expansion

        # Need projection if spatial size or channel count changes
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [Bottleneck(in_channels, planes, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward(self, x):
        # Stem
        x  = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)   # [B, 256,  H/4,  W/4]
        c2 = self.layer2(c1)  # [B, 512,  H/8,  W/8]
        c3 = self.layer3(c2)  # [B, 1024, H/16, W/16]
        c4 = self.layer4(c3)  # [B, 2048, H/32, W/32]
        return [c1, c2, c3, c4]


# -------------------------
# UPerNet Decoder
# -------------------------

class PPM(nn.Module):
    """
    Pyramid Pooling Module.
    Pools C4 at 4 scales, upsamples back, concatenates, and fuses.
    Captures global context before FPN top-down pathway.
    """
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
        h, w = x.shape[2], x.shape[3]
        pooled = []
        for stage in self.stages:
            # MPS does not support non-divisible adaptive pool — fall back to CPU
            if x.device.type == "mps":
                out = stage(x.cpu()).to(x.device)
            else:
                out = stage(x)
            pooled.append(
                F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat([x] + pooled, dim=1))


class UPerNetHead(nn.Module):
    """
    UPerNet segmentation head.

    Steps:
        1. PPM on C4 for global context
        2. Lateral 1x1 convs to unify channel dims → fpn_dim
        3. Top-down FPN: C4 -> C3 -> C2 -> C1
        4. 3x3 smooth conv on each FPN level
        5. Upsample all to C1 resolution, concatenate, fuse
        6. 1x1 conv → num_classes logits

    Args:
        in_channels (list[int]): [C1_ch, C2_ch, C3_ch, C4_ch] from encoder
        num_classes (int)
        fpn_dim (int): internal channel dimension (default 256)
        dropout (float)
    """
    def __init__(self, in_channels, num_classes, fpn_dim=256, dropout=0.1):
        super().__init__()
        assert len(in_channels) == 4

        self.ppm = PPM(in_channels[3])

        # Lateral 1x1 convs (unify all encoder channels to fpn_dim)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ) for c in in_channels
        ])

        # Smooth 3x3 convs after top-down addition
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ) for _ in in_channels
        ])

        # Fuse all 4 FPN levels (concat → single feature map)
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

        # PPM on deepest feature
        c4 = self.ppm(c4)

        # Lateral projections
        p1 = self.laterals[0](c1)
        p2 = self.laterals[1](c2)
        p3 = self.laterals[2](c3)
        p4 = self.laterals[3](c4)

        # Top-down FPN
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode="bilinear", align_corners=False)
        p2 = p2 + F.interpolate(p3, size=p2.shape[2:], mode="bilinear", align_corners=False)
        p1 = p1 + F.interpolate(p2, size=p1.shape[2:], mode="bilinear", align_corners=False)

        # Smooth each level
        p1 = self.smooth[0](p1)
        p2 = self.smooth[1](p2)
        p3 = self.smooth[2](p3)
        p4 = self.smooth[3](p4)

        # Upsample all to p1 resolution
        target = p1.shape[2:]
        p2 = F.interpolate(p2, size=target, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=target, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=target, mode="bilinear", align_corners=False)

        # Fuse and classify
        fused  = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))
        logits = self.classifier(fused)
        return logits


# -------------------------
# Full Model
# -------------------------

class ResNet101UPerNet(nn.Module):
    """
    ResNet-101 encoder + UPerNet decoder for semantic segmentation.

    Args:
        num_classes (int): number of segmentation classes
        fpn_dim (int): UPerNet internal feature dimension
        dropout (float): dropout in fusion layer
    """
    def __init__(self, num_classes=12, fpn_dim=256, dropout=0.1):
        super().__init__()
        self.encoder    = ResNet101Encoder()
        self.decode_head = UPerNetHead(
            in_channels=self.encoder.out_channels,
            num_classes=num_classes,
            fpn_dim=fpn_dim,
            dropout=dropout,
        )

    def forward(self, x):
        feats  = self.encoder(x)           # [c1, c2, c3, c4]
        logits = self.decode_head(feats)   # [B, num_classes, H/4, W/4]
        logits = F.interpolate(logits, size=x.shape[2:],
                               mode="bilinear", align_corners=False)
        return logits
