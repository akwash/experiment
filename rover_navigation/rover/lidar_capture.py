"""
Capture one LiDAR scan and save it as CSV for inference.

Usage:
    python -m rover_navigation.rover.lidar_capture --out rover_navigation/data/in_situ_scans

The output CSV is compatible with `rover_navigation.rover.csv_loader.load_csv_point_cloud`
because it contains at least:
    - X1_mm
    - Y1_mm
    - Z1_mm

Optionally, a timestamp_ns column can also be included.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from datetime import datetime
import csv
from pathlib import Path
import time
from typing import Iterable

import numpy as np


class LidarCaptureError(RuntimeError):
    """Raised when a LiDAR scan cannot be captured or saved."""


class BaseLidarReader(ABC):
    """
    Hardware-specific LiDAR interface.

    Implementations must return one full scan as an array with shape (N, 3),
    where columns are [X1_mm, Y1_mm, Z1_mm] in the LiDAR local frame.
    """

    @abstractmethod
    def capture_one_scan_mm(self) -> np.ndarray: 
        """Capture one scan and return points as (N, 3) in millimeters."""


@dataclass
class RPLidarReader(BaseLidarReader):
    """
    Reader for 2D RPLidar devices.

    Notes:
    - Produces XY points from polar samples and sets Z1_mm=0.
    - Requires the optional `rplidar` package on the Jetson.
    """

    port: str 
    timeout_s: float = 8.0 # max seconds to wait for scan
    min_points: int = 30 # minimum points from a scan to be valid

    def capture_one_scan_mm(self) -> np.ndarray: 
        """
        Capture one scan get points in (N,3) in mm
        :return: points as (N,3)
        """
        try:
            from rplidar import RPLidar  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on Jetson environment
            raise LidarCaptureError(
                "Failed to import rplidar. Install it on the Jetson first."
            ) from exc

        lidar = None 
        start_t = time.monotonic() # start time to track timeout
        try:
            lidar = RPLidar(self.port) # connect to LiDAR on port

            for scan in lidar.iter_scans():
                if time.monotonic() - start_t > self.timeout_s:
                    raise LidarCaptureError(
                        f"Timed out waiting for scan after {self.timeout_s:.1f}s."
                    )

                if not scan:
                    continue

                points = []
                for _quality, angle_deg, distance_mm in scan:
                    if distance_mm <= 0:
                        continue
                    theta = np.deg2rad(angle_deg) # convert angle to rad
                    x_mm = float(distance_mm * np.cos(theta)) # convert polar to cartesian
                    y_mm = float(distance_mm * np.sin(theta)) # convert polar to cartesian
                    points.append((x_mm, y_mm, 0.0)) # set z = 0

                if len(points) >= self.min_points:
                    return np.asarray(points, dtype=np.float32) # return points as (N,3)

            raise LidarCaptureError("LiDAR returned no usable scans.")
        except LidarCaptureError:
            raise
        except Exception as exc:  # pragma: no cover - depends on hardware
            raise LidarCaptureError(f"Unexpected LiDAR read failure: {exc}") from exc
        finally:
            if lidar is not None: # LiDAR cleanup (disconnect)
                try:
                    lidar.stop()
                except Exception:
                    pass
                try:
                    lidar.stop_motor()
                except Exception:
                    pass
                try:
                    lidar.disconnect()
                except Exception:
                    pass


def write_scan_csv(
    points_mm: np.ndarray,
    out_csv: Path,
    include_timestamp_ns: bool = True,
    timestamp_ns: int | None = None,
) -> None:
    """
    Save LiDAR points to CSV with headers required by csv_loader.py.

    :param points_mm: (N, 3) [X1_mm, Y1_mm, Z1_mm]
    :param out_csv: output file path
    :param include_timestamp_ns: include timestamp_ns column if True
    :param timestamp_ns: value to write in timestamp_ns (defaults to current time)
    """
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        raise LidarCaptureError(f"Expected points shape (N, 3), got {points_mm.shape}")
    if points_mm.shape[0] == 0:
        raise LidarCaptureError("Refusing to write empty scan.")

    out_csv.parent.mkdir(parents=True, exist_ok=True) # check for output directory
    timestamp_ns = int(time.time_ns() if timestamp_ns is None else timestamp_ns) # set current time

    fieldnames = ["X1_mm", "Y1_mm", "Z1_mm"] # requried columns for csv_loader.py
    if include_timestamp_ns:
        fieldnames.append("timestamp_ns") 

    try:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames) # create CSV writer
            writer.writeheader()

            for x_mm, y_mm, z_mm in points_mm:
                row = {
                    "X1_mm": float(x_mm),
                    "Y1_mm": float(y_mm),
                    "Z1_mm": float(z_mm),
                }
                if include_timestamp_ns:
                    row["timestamp_ns"] = timestamp_ns
                writer.writerow(row)
    except OSError as exc:
        raise LidarCaptureError(f"Failed writing CSV: {out_csv}") from exc


def default_scan_filename() -> str:
    """Build a timestamped filename for one scan."""
    utc = datetime.utcnow().strftime("%Y%m%d_%H%M%S") # UTC timestamp
    return f"scan_{utc}_{time.time_ns()}.csv"


def capture_and_save_one_scan(
    reader: BaseLidarReader,
    out_dir: Path,
    include_timestamp_ns: bool = True,
) -> Path:
    """Capture one scan with the provided reader and save one CSV file.
    :param reader: LiDAR reader
    :param out_dir: directory to save CSV
    :param include_timestamp_ns: include timestamp column
    :return: path to csv
    """
    points_mm = reader.capture_one_scan_mm() # points (N,3) in mm
    out_csv = out_dir / default_scan_filename() # output path
    write_scan_csv(
        points_mm=points_mm,
        out_csv=out_csv,
        include_timestamp_ns=include_timestamp_ns,
    )
    return out_csv


def build_reader(args: argparse.Namespace) -> BaseLidarReader:
    """
    Create a hardware reader from CLI arguments.
    :param args: CLI arguments
    :return: LiDAR reader instance
    """
    if args.backend == "rplidar":
        if not args.port:
            raise LidarCaptureError("--port is required when backend is 'rplidar'.")
        return RPLidarReader(
            port=args.port,
            timeout_s=args.timeout_s,
            min_points=args.min_points,
        )

    raise LidarCaptureError(f"Unsupported backend: {args.backend}")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for LiDAR capture.
    """
    parser = argparse.ArgumentParser(
        description="Capture one LiDAR scan and save it as CSV."
    )
    parser.add_argument(
        "--out",
        default="rover_navigation/data/in_situ_scans",
        help="Output directory for captured scan CSV files.",
    )
    parser.add_argument(
        "--backend",
        default="rplidar",
        choices=["rplidar"],
        help="LiDAR backend reader.",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port for RPLidar (example: /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=8.0,
        help="Maximum seconds to wait for one scan.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=30,
        help="Minimum points required to accept a scan.",
    )
    parser.add_argument(
        "--no-timestamp-ns",
        action="store_true",
        help="Do not include timestamp_ns column in CSV.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser() # bukld CLI parser
    args = parser.parse_args(list(argv) if argv is not None else None) # parse CLI

    try:
        reader = build_reader(args) # create LiDAR reader
        out_dir = Path(args.out) # output directory
        out_csv = capture_and_save_one_scan(
            reader=reader,
            out_dir=out_dir,
            include_timestamp_ns=not args.no_timestamp_ns,
        )
        print(f"Saved scan to: {out_csv}")
        return 0
    except LidarCaptureError as exc:
        print(f"LiDAR capture error: {exc}")
        return 2
    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
