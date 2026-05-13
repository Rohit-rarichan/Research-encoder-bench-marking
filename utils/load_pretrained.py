"""
Loads nvidia/segformer-b0-finetuned-ade-512-512 weights into SegformerClasswise.

Add load_pretrained() as a method on SegformerClasswise, or just call it standalone.

Usage:
    from load_pretrained import load_pretrained_hf
    model = SegformerClasswise(num_classes=150)
    load_pretrained_hf(model, "nvidia/segformer-b0-finetuned-ade-512-512")
"""

import torch
from transformers import SegformerForSemanticSegmentation


def load_pretrained_hf(model, hf_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
    """
    Transfers weights from a HuggingFace SegFormer checkpoint into a
    SegformerClasswise instance. Returns a list of any keys that could
    not be mapped.
    """
    hf = SegformerForSemanticSegmentation.from_pretrained(hf_id)
    hf_sd = hf.state_dict()

    my_sd = model.state_dict()
    new_sd = {}
    unmapped = []

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    def copy(my_key, hf_key):
        if hf_key not in hf_sd:
            unmapped.append(f"MISSING in HF : {hf_key}")
            return
        if my_key not in my_sd:
            unmapped.append(f"MISSING in MY : {my_key}")
            return
        h, m = hf_sd[hf_key], my_sd[my_key]
        if h.shape != m.shape:
            unmapped.append(f"SHAPE MISMATCH {my_key}: hf={tuple(h.shape)} my={tuple(m.shape)}")
            return
        new_sd[my_key] = h

    # ------------------------------------------------------------------ #
    # Encoder — patch embeddings
    # ------------------------------------------------------------------ #
    for i in range(4):
        pfx_hf = f"segformer.encoder.patch_embeddings.{i}"
        pfx_my = f"encoder.patch_embeds.{i}"
        copy(f"{pfx_my}.proj.weight",   f"{pfx_hf}.proj.weight")
        copy(f"{pfx_my}.proj.bias",     f"{pfx_hf}.proj.bias")
        copy(f"{pfx_my}.norm.weight",   f"{pfx_hf}.layer_norm.weight")
        copy(f"{pfx_my}.norm.bias",     f"{pfx_hf}.layer_norm.bias")

    # ------------------------------------------------------------------ #
    # Encoder — transformer blocks
    # ------------------------------------------------------------------ #
    depths = (2, 2, 2, 2)
    for i in range(4):
        for j in range(depths[i]):
            pfx_hf = f"segformer.encoder.block.{i}.{j}"
            pfx_my = f"encoder.stages.{i}.{j}"

            # layer norms
            copy(f"{pfx_my}.norm1.weight", f"{pfx_hf}.layer_norm_1.weight")
            copy(f"{pfx_my}.norm1.bias",   f"{pfx_hf}.layer_norm_1.bias")
            copy(f"{pfx_my}.norm2.weight", f"{pfx_hf}.layer_norm_2.weight")
            copy(f"{pfx_my}.norm2.bias",   f"{pfx_hf}.layer_norm_2.bias")

            # attention — Q
            copy(f"{pfx_my}.attn.q.weight", f"{pfx_hf}.attention.self.query.weight")
            copy(f"{pfx_my}.attn.q.bias",   f"{pfx_hf}.attention.self.query.bias")

            # attention — KV (HF stores K and V separately; we concatenate)
            k_w = hf_sd[f"{pfx_hf}.attention.self.key.weight"]
            v_w = hf_sd[f"{pfx_hf}.attention.self.value.weight"]
            k_b = hf_sd[f"{pfx_hf}.attention.self.key.bias"]
            v_b = hf_sd[f"{pfx_hf}.attention.self.value.bias"]
            new_sd[f"{pfx_my}.attn.kv.weight"] = torch.cat([k_w, v_w], dim=0)
            new_sd[f"{pfx_my}.attn.kv.bias"]   = torch.cat([k_b, v_b], dim=0)

            # attention — output projection
            copy(f"{pfx_my}.attn.proj.weight", f"{pfx_hf}.attention.output.dense.weight")
            copy(f"{pfx_my}.attn.proj.bias",   f"{pfx_hf}.attention.output.dense.bias")

            # attention — sequence reduction (sr) conv + norm
            sr_key = f"{pfx_hf}.attention.self.sr.weight"
            if sr_key in hf_sd:
                copy(f"{pfx_my}.attn.sr.weight",   f"{pfx_hf}.attention.self.sr.weight")
                copy(f"{pfx_my}.attn.sr.bias",     f"{pfx_hf}.attention.self.sr.bias")
                copy(f"{pfx_my}.attn.norm.weight", f"{pfx_hf}.attention.self.layer_norm.weight")
                copy(f"{pfx_my}.attn.norm.bias",   f"{pfx_hf}.attention.self.layer_norm.bias")

            # MLP / Mix-FFN
            copy(f"{pfx_my}.mlp.fc1.weight",    f"{pfx_hf}.mlp.dense1.weight")
            copy(f"{pfx_my}.mlp.fc1.bias",      f"{pfx_hf}.mlp.dense1.bias")
            copy(f"{pfx_my}.mlp.fc2.weight",    f"{pfx_hf}.mlp.dense2.weight")
            copy(f"{pfx_my}.mlp.fc2.bias",      f"{pfx_hf}.mlp.dense2.bias")
            copy(f"{pfx_my}.mlp.dwconv.weight", f"{pfx_hf}.mlp.dwconv.dwconv.weight")
            copy(f"{pfx_my}.mlp.dwconv.bias",   f"{pfx_hf}.mlp.dwconv.dwconv.bias")

    # ------------------------------------------------------------------ #
    # Encoder — stage layer norms
    # ------------------------------------------------------------------ #
    for i in range(4):
        copy(f"encoder.stage_norms.{i}.weight", f"segformer.encoder.layer_norm.{i}.weight")
        copy(f"encoder.stage_norms.{i}.bias",   f"segformer.encoder.layer_norm.{i}.bias")

    # ------------------------------------------------------------------ #
    # Decoder — per-scale MLP projections
    # ------------------------------------------------------------------ #
    for i in range(4):
        copy(f"decode_head.mlp_projections.{i}.proj.weight",
             f"decode_head.linear_c.{i}.proj.weight")
        copy(f"decode_head.mlp_projections.{i}.proj.bias",
             f"decode_head.linear_c.{i}.proj.bias")

    # ------------------------------------------------------------------ #
    # Decoder — fusion layer
    # HF uses Conv2d(kernel=1) → weight shape [C_out, C_in, 1, 1]
    # Our fuse_mlp[0] uses nn.Linear → weight shape [C_out, C_in]
    # ------------------------------------------------------------------ #
    fuse_w_hf = hf_sd["decode_head.linear_fuse.weight"]
    new_sd["decode_head.fuse_mlp.0.proj.weight"] = fuse_w_hf.squeeze(-1).squeeze(-1)

    # BatchNorm after fuse
    copy("decode_head.fuse_mlp.1.weight",       "decode_head.batch_norm.weight")
    copy("decode_head.fuse_mlp.1.bias",         "decode_head.batch_norm.bias")
    copy("decode_head.fuse_mlp.1.running_mean", "decode_head.batch_norm.running_mean")
    copy("decode_head.fuse_mlp.1.running_var",  "decode_head.batch_norm.running_var")
    copy("decode_head.fuse_mlp.1.num_batches_tracked",
         "decode_head.batch_norm.num_batches_tracked")

    # ------------------------------------------------------------------ #
    # Decoder — classifier
    # Commented out so custom class counts (e.g. 7 classes) keep a random
    # classifier head instead of trying to load the 150-class ADE head.
    # ------------------------------------------------------------------ #
    # copy("decode_head.classifier.weight", "decode_head.classifier.weight")
    # copy("decode_head.classifier.bias",   "decode_head.classifier.bias")

    # ------------------------------------------------------------------ #
    # Load into model
    # ------------------------------------------------------------------ #
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    print(f"\n=== Weight transfer complete ===")
    print(f"Mapped     : {len(new_sd)} tensors")
    print(f"Unmapped   : {len(unmapped)}")
    for u in unmapped:
        print("  ", u)
    print(f"Missing in new_sd (kept random): {missing}")
    print(f"Unexpected keys (ignored)      : {unexpected}")
    return unmapped