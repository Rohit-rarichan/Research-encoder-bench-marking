import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x / keep_prob * random_tensor


# -------------------------
# Overlap Patch Embedding
# -------------------------
class OverlapPatchEmbed(nn.Module):
    """
    Conv2d -> flatten -> LayerNorm
    Produces tokens [B, N, C] and spatial size (H, W)
    """
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: int, 
                 layer_norm_eps : float = 1e-6):
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim,eps = layer_norm_eps)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)              # [B, embed_dim, H', W']
        H, W = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        return x, H, W


# -------------------------
# MLP / Mix-FFN
# -------------------------
class MixFFN(nn.Module):
    """
    Linear -> depthwise conv (in feature map space) -> GELU -> Linear
    """
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # x: [B, N, C]
        B, N, C = x.shape
        x = self.fc1(x)   # [B, N, hidden]
        x = self.drop(x)

        # reshape to feature map for depthwise conv
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # [B, hidden, H, W]
        x = self.dwconv(x)
        x = x.reshape(B, -1, N).transpose(1, 2)     # back to [B, N, hidden]

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------
# Efficient Self-Attention with Sequence Reduction
# -------------------------
class SREfficientSelfAttention(nn.Module):
    """
    Q from full tokens, K/V from reduced tokens via strided Conv2d
    """
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, attn_drop: float = 0.0,
                  proj_drop: float = 0.0, layer_norm_eps : float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim,eps=layer_norm_eps)
        else:
            self.sr = None
            self.norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,h,N,hd]

        if self.sr_ratio > 1:
            # reduce tokens for K/V
            x_ = x.transpose(1, 2).reshape(B, C, H, W)  # [B,C,H,W]
            x_ = self.sr(x_)                            # [B,C,H',W']
            x_ = x_.reshape(B, C, -1).transpose(1, 2)   # [B,N',C]
            x_ = self.norm(x_)
            kv = self.kv(x_)
            N_kv = x_.shape[1]
        else:
            kv = self.kv(x)
            N_kv = N

        kv = kv.reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B,h,N',hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,N,N']
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,h,N,hd]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B,N,C]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------
# Transformer Block
# -------------------------
class SegformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sr_ratio: int, 
                 dropout: float = 0.0, attn_drop : float = 0.0, 
                 drop_path_prob:float = 0.0, layer_norm_eps : float = 1e-6):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim,eps=layer_norm_eps)
        self.attn = SREfficientSelfAttention(dim, num_heads=num_heads, 
                                             sr_ratio=sr_ratio, attn_drop=attn_drop, proj_drop=dropout,
                                             layer_norm_eps=layer_norm_eps)

        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = MixFFN(dim, hidden_dim, drop=dropout)
        self.drop_path = DropPath(drop_path_prob) 


    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# -------------------------
# Encoder (4 stages)
# -------------------------
class SegformerEncoder(nn.Module):
    """
    Builds the hierarchical encoder.
    """
    def __init__(
        self,
        in_chans=3,
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4.0, 4.0, 4.0, 4.0),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        dropout : float = 0.0,
        attn_drop : float = 0.0,
        drop_path_rate : float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        # patch embeddings per stage (patch_size, stride) match SegFormer family
        #note: Instead of invoking it 4 times, make it such that you can provide it with a 
        #input tensor and all the dimensions
        #modify overlappactchembed for 4 conv layers
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(in_chans, embed_dims[0], patch_size=7, stride=4, layer_norm_eps=layer_norm_eps),
            OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2, layer_norm_eps=layer_norm_eps),
            OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2, layer_norm_eps=layer_norm_eps),
            OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2, layer_norm_eps=layer_norm_eps),
        ])

        #figure out what this does
        total_blocks = int(sum(depths))
        if total_blocks > 0:
            dpr = torch.linspace(0.0, float(drop_path_rate), total_blocks).tolist()
        else:
            dpr = []
        cur = 0

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = nn.ModuleList([
                SegformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    sr_ratio=sr_ratios[i],
                    dropout=dropout,
                    attn_drop=attn_drop,
                    drop_path_prob=dpr[cur + j] if dpr else 0.0,
                    layer_norm_eps=layer_norm_eps,
                )
                for j in range(depths[i])
            ])
            self.stages.append(blocks)
            cur += depths[i]

        self.stage_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dims[i], eps=layer_norm_eps) for i in range(4)])

    def forward(self, x):
        # returns feature maps at 4 scales: [B,C_i,H_i,W_i]
        outs = []
        for stage_idx in range(4):
            x, H, W = self.patch_embeds[stage_idx](x)
            for blk in self.stages[stage_idx]:
                x = blk(x, H, W)
            # tokens -> feature map
            x = self.stage_norms[stage_idx](x)  
            B, N, C = x.shape
            feat = x.transpose(1, 2).reshape(B, C, H, W)
            outs.append(feat)
            x = feat  # next stage expects [B,C,H,W]
        return outs

# -------------------------
# Decoder Head (SegFormer Head)
# -------------------------
class SegformerMLP(nn.Module):
    """Linear projection on flattened spatial tokens, as in the paper."""
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias = bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]  →  flatten spatial dims  →  project  →  restore
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # [B, H*W, C]
        x = self.proj(x)                   # [B, H*W, decoder_dim]
        x = x.transpose(1, 2).reshape(B, -1, H, W)  # [B, decoder_dim, H, W]
        return x


class SegformerDecoderHead(nn.Module):
    """
    Takes 4 multi-scale feature maps from encoder:
      feats = [c1, c2, c3, c4]
    Projects each to decoder_dim via MLP, upsamples to c1 spatial size,
    concatenates, fuses with MLP, and outputs segmentation logits.
    """
    def __init__(self, in_channels=(32, 64, 160, 256), decoder_dim=256, num_classes=150, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.decoder_dim = decoder_dim

        # Per-scale MLP projections (Linear, not Conv2d)
        self.mlp_projections = nn.ModuleList([
            SegformerMLP(in_ch, decoder_dim) for in_ch in in_channels
        ])

        # Fusion MLP: takes concatenated [4 * decoder_dim] channels → decoder_dim
        # Implemented as Linear over the channel dim (equivalent to pointwise)
        self.fuse_mlp = nn.Sequential(
            SegformerMLP(decoder_dim * 4, decoder_dim, bias = False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Final pixel-wise classifier
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        target_h, target_w = c1.shape[-2], c1.shape[-1]

        # 1. MLP projection + upsample each scale to c1 resolution
        projected = []
        for feat, mlp in zip([c1, c2, c3, c4], self.mlp_projections):
            p = mlp(feat)  # [B, decoder_dim, Hi, Wi]
            p = F.interpolate(p, size=(target_h, target_w), mode="bilinear", align_corners=False)
            projected.append(p)

        # 2. Concatenate along channel dim → [B, 4*decoder_dim, H/4, W/4]
        x = torch.cat(projected[::-1], dim=1)

        # 3. Fuse with MLP → [B, decoder_dim, H/4, W/4]
        x = self.fuse_mlp(x)

        # 4. Classify → [B, num_classes, H/4, W/4]
        logits = self.classifier(x)
        return logits

# -------------------------
# Full Model = Encoder + Decoder
# -------------------------
class SegformerClasswise(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=150,
        # Encoder config (this is SegFormer-B0 by default)
        embed_dims=(32, 64, 160, 256),    #also called our ouput channels
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4.0, 4.0, 4.0, 4.0),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        dropout: float = 0.0,          # hidden_dropout_prob (MLP + proj)
        attn_drop: float = 0.0,        # attention_probs_dropout_prob
        drop_path_rate: float = 0.0,   # stochastic depth max
        layer_norm_eps: float = 1e-6,
        # Decoder config
        decoder_dim=256,
        decoder_dropout=0.1,
    ):
        super().__init__()
        self.encoder = SegformerEncoder(
            in_chans=in_chans,
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
        self.decode_head = SegformerDecoderHead(
            in_channels=embed_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
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
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        feats = self.encoder(x)          # list of 4 feature maps
        logits = self.decode_head(feats) # [B, num_classes, H1, W1]
        return logits