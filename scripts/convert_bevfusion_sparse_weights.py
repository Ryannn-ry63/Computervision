#!/usr/bin/env python
"""Convert BEVFusion sparse conv weights to the layout used by this install."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    converted = []
    for key, value in list(state_dict.items()):
        if key.startswith("pts_middle_encoder") and getattr(value, "ndim", 0) == 5:
            # Saved layout: [out_channels, kD, kH, kW, in_channels].
            # Current sparse conv layout: [kD, kH, kW, in_channels, out_channels].
            state_dict[key] = value.permute(1, 2, 3, 4, 0).contiguous()
            converted.append(key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Converted {len(converted)} sparse conv weights.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
