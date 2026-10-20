#!/usr/bin/env python3
"""Export the inference-only ViT encoder from a full MAE training checkpoint."""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.project_root))
    from models_vit import vit_batter_patch16

    print(f"Loading {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True)
    model = vit_batter_patch16(img_size=128, in_chans=6)
    load_result = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [
        key
        for key in load_result.unexpected_keys
        if not key.startswith("decoder_") and key != "mask_token"
    ]
    if load_result.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch; missing={load_result.missing_keys}, unexpected={unexpected}"
        )
    del checkpoint

    artifact = {
        "format_version": 1,
        "model_name": "vit_batter_patch16",
        "model_kwargs": {"img_size": 128, "in_chans": 6},
        "storage_dtype": "float16",
        "state_dict": {
            key: value.detach().cpu().half() for key, value in model.state_dict().items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
