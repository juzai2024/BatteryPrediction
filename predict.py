#!/usr/bin/env python3
"""Predict battery capacity from one or more six-channel MAT files."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from battery_model import CapacityHead, vit_battery_patch16


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENCODER = PROJECT_ROOT / "weights" / "battery_vit_encoder_fp16.pth"


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_torch_artifact(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_models(battery, encoder_path, head_path, device):
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Encoder weight not found: {encoder_path}\n"
            "Run `python download_weights.py` first."
        )
    if not head_path.exists():
        raise FileNotFoundError(f"Capacity-head weight not found: {head_path}")

    encoder_artifact = load_torch_artifact(encoder_path)
    encoder = vit_battery_patch16(**encoder_artifact["model_kwargs"])
    encoder.load_state_dict(encoder_artifact["state_dict"], strict=True)
    encoder.eval().to(device)

    head_artifact = load_torch_artifact(head_path)
    if head_artifact["battery_type"].upper() != battery:
        raise ValueError(
            f"Head {head_path} is for {head_artifact['battery_type']}, not {battery}."
        )
    head = CapacityHead(**head_artifact["model_kwargs"])
    head.load_state_dict(head_artifact["state_dict"], strict=True)
    head.eval().to(device)
    return encoder, head


def collect_inputs(input_path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.mat"))
        if paths:
            return paths
        raise FileNotFoundError(f"No MAT files found in {input_path}")
    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_sample(path):
    sample = sio.loadmat(path)
    if "data" not in sample:
        raise KeyError(f"{path} does not contain the required `data` array.")
    image = np.asarray(sample["data"], dtype=np.float32)
    if image.shape != (128, 128, 6):
        raise ValueError(
            f"Expected `data` shape (128, 128, 6), got {image.shape}: {path}"
        )
    if not np.isfinite(image).all():
        raise ValueError(f"Input contains NaN or infinite values: {path}")
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())

    true_capacity = None
    if "ah" in sample:
        label = np.asarray(sample["ah"]).reshape(-1)
        if label.size:
            true_capacity = float(label[0])
    return image_tensor, true_capacity


def predict(paths, encoder, head, device, batch_size):
    results = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            loaded = [load_sample(path) for path in batch_paths]
            images = torch.stack([item[0] for item in loaded]).to(device)
            features = encoder(images)
            capacities = head(features)[0].reshape(-1).cpu().numpy()
            for path, (_, true_capacity), capacity in zip(
                batch_paths, loaded, capacities
            ):
                result = {
                    "file": str(path),
                    "predicted_capacity_ah": float(capacity),
                }
                if true_capacity is not None:
                    result["true_capacity_ah"] = true_capacity
                    result["absolute_error_ah"] = abs(float(capacity) - true_capacity)
                results.append(result)
    return results


def print_results(results):
    has_labels = all("true_capacity_ah" in result for result in results)
    if has_labels:
        print(f"{'file':44} {'true (Ah)':>10} {'pred (Ah)':>10} {'abs err':>10}")
        for result in results:
            print(
                f"{Path(result['file']).name:44} "
                f"{result['true_capacity_ah']:10.4f} "
                f"{result['predicted_capacity_ah']:10.4f} "
                f"{result['absolute_error_ah']:10.4f}"
            )
    else:
        print(f"{'file':44} {'predicted capacity (Ah)':>24}")
        for result in results:
            print(
                f"{Path(result['file']).name:44} "
                f"{result['predicted_capacity_ah']:24.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", required=True, choices=["LCO", "NMC"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    battery = args.battery.upper()
    input_path = args.input or PROJECT_ROOT / "toy_data" / battery
    head_path = args.head or PROJECT_ROOT / "weights" / f"{battery.lower()}_capacity_head.pth"
    device = choose_device(args.device)
    paths = collect_inputs(input_path)
    encoder, head = load_models(
        battery=battery,
        encoder_path=args.encoder,
        head_path=head_path,
        device=device,
    )
    results = predict(paths, encoder, head, device, args.batch_size)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Device: {device}")
        print_results(results)


if __name__ == "__main__":
    main()
