from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def convert_bin_to_ply(input_path: Path, output_path: Path) -> None:
    cloudcompare = shutil.which("CloudCompare")
    if cloudcompare is None:
        raise RuntimeError(
            "CloudCompare executable not found in PATH. Install CloudCompare CLI first."
        )

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        cloudcompare,
        "-SILENT",
        "-O",
        str(input_path),
        "-C_EXPORT_FMT",
        "PLY",
        "-SAVE_CLOUDS",
        "FILE",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CloudCompare .bin to .ply")
    parser.add_argument("--input", required=True, type=Path, help="Path to input .bin")
    parser.add_argument("--output", required=True, type=Path, help="Path to output .ply")
    args = parser.parse_args()

    if args.input.suffix.lower() != ".bin":
        raise ValueError(f"Expected .bin input, got: {args.input}")
    if args.output.suffix.lower() != ".ply":
        raise ValueError(f"Expected .ply output, got: {args.output}")

    convert_bin_to_ply(args.input, args.output)


if __name__ == "__main__":
    main()
