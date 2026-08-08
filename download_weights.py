#!/usr/bin/env python3
"""Download and verify the shared ViT encoder weight from GitHub Releases."""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FILENAME = "battery_vit_encoder_fp16.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "weights" / FILENAME
DOWNLOAD_URL = (
    "https://github.com/juzai2024/BatteryPrediction/releases/download/"
    f"v0.1.0/{FILENAME}"
)
EXPECTED_SHA256 = "f41e883661a1850832af2488dee2630ff532dda54bab1f62ce2545aaee0e4034"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        if sha256(args.output) == EXPECTED_SHA256:
            print(f"Encoder weight already exists and is valid: {args.output}")
            return
        raise RuntimeError(
            f"Existing file has the wrong checksum: {args.output}. "
            "Use --force to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    print(f"Downloading {DOWNLOAD_URL}")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, partial)
        actual_sha256 = sha256(partial)
        if actual_sha256 != EXPECTED_SHA256:
            raise RuntimeError(
                f"Checksum mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
            )
        partial.replace(args.output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"Saved and verified: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Download cancelled.", file=sys.stderr)
        raise SystemExit(130)
