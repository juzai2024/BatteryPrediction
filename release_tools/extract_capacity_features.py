#!/usr/bin/env python3
"""Extract frozen ViT features for battery-capacity regression."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


class MatDataset(Dataset):
    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample = sio.loadmat(self.paths[index])
        image = np.asarray(sample["data"], dtype=np.float32)
        if image.shape != (128, 128, 6):
            raise ValueError(f"Unexpected data shape {image.shape}: {self.paths[index]}")
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        capacity = float(np.asarray(sample["ah"]).reshape(-1)[0])
        return image, capacity, str(self.paths[index])


def parse_dataset(value):
    try:
        battery, config = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected BATTERY=/path/to/config.json") from exc
    return battery.upper(), Path(config)


def extract(model, paths, device, batch_size, workers):
    loader = DataLoader(
        MatDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    features, labels, ordered_paths = [], [], []
    with torch.inference_mode():
        for images, capacities, batch_paths in loader:
            output = model(images.to(device, non_blocking=True))
            features.append(output.float().cpu().numpy())
            labels.append(capacities.float().numpy())
            ordered_paths.extend(batch_paths)
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.asarray(ordered_paths),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(args.project_root))
    from models_vit import vit_batter_patch16

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = vit_batter_patch16(img_size=128, in_chans=6)
    load_result = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [key for key in load_result.unexpected_keys if not key.startswith("decoder_") and key != "mask_token"]
    if load_result.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch; missing={load_result.missing_keys}, unexpected={unexpected}"
        )
    del checkpoint
    model.eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for battery, config_path in args.dataset:
        config = json.load(config_path.open())[battery]
        print(
            f"Extracting {battery}: train={len(config['train'])}, test={len(config['test'])}",
            flush=True,
        )
        arrays = {}
        for split in ("train", "test"):
            x, y, paths = extract(
                model,
                config[split],
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
            )
            arrays[f"x_{split}"] = x
            arrays[f"y_{split}"] = y
            arrays[f"paths_{split}"] = paths
        output_path = args.output_dir / f"{battery.lower()}_features.npz"
        np.savez_compressed(output_path, **arrays)
        print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
