from __future__ import annotations

import numpy as np


def get_latest_scan_xyz() -> np.ndarray:
    """
    Example hook for `scan_mode: function`.

    Replace this body with your real LiDAR acquisition path and return one scan
    as an `(N, 3)` numpy array in sensor frame coordinates.
    """
    # Placeholder example: returns a tiny synthetic scan.
    # Keep this while wiring hardware, then replace with real source.
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [1.2, 0.2, 0.0],
            [1.2, -0.2, 0.0],
        ],
        dtype=np.float32,
    )

