"""
Phase 1 deterministic test-window split (jrn_01KQPNN9D9GQ113CJ0Y3R83PAZ).

70% of WADI's 16 anomaly time-ranges → policy-training set (BC + PPO).
30% → held-out evaluation set (DP2 multi-metric).

Split is by range index, not by window — all windows within a single anomaly
range stay in the same partition. This preserves per-attack semantics while
preventing distribution leakage from test-data structure into the policy's
training signal.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, NamedTuple

import numpy as np

from loaders.wadi_loader import WADI_ATTACK_RANGES, WADI_ATTACK_BASE


class AnomalyRange(NamedTuple):
    """A WADI test anomaly time-range, mapped to downsampled-row coordinates."""
    range_index: int  # position in WADI_ATTACK_RANGES (0-based)
    start_row_raw: int  # 1Hz row index (inclusive)
    end_row_raw: int  # 1Hz row index (exclusive)
    start_row_ds: int  # downsample-by-10 row index (inclusive)
    end_row_ds: int  # downsample-by-10 row index (exclusive)


def _parse(date_str: str, time_str: str) -> datetime:
    return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S")


def get_wadi_anomaly_ranges_ds(num_test_rows_ds: int = 17280) -> list[AnomalyRange]:
    """Return per-range coordinates in downsampled test-row coordinates.

    The wadi_loader's `derive_attack_labels` works on raw 1Hz rows; the
    downsampled test data uses every 10th row aggregated. We mirror that mapping.
    """
    ranges = []
    for i, (start_d, start_t, end_d, end_t) in enumerate(WADI_ATTACK_RANGES):
        start_sec = int((_parse(start_d, start_t) - WADI_ATTACK_BASE).total_seconds())
        end_sec = int((_parse(end_d, end_t) - WADI_ATTACK_BASE).total_seconds())
        s_raw = max(start_sec, 0)
        e_raw = min(end_sec + 1, num_test_rows_ds * 10)
        if s_raw >= e_raw:
            # Skip degenerate / typo ranges (e.g., the 10/11 → 10/10 inversion at index 9).
            continue
        s_ds = s_raw // 10
        e_ds = e_raw // 10 + 1
        e_ds = min(e_ds, num_test_rows_ds)
        ranges.append(AnomalyRange(i, s_raw, e_raw, s_ds, e_ds))
    return ranges


def split_ranges(ranges: list[AnomalyRange], train_fraction: float = 0.7) -> tuple[list[AnomalyRange], list[AnomalyRange]]:
    """Deterministic split by range index, not random.

    First `floor(train_fraction * N)` ranges → train; remainder → eval.
    With 16 candidate ranges and 70% threshold, 11 train / 5 holdout.
    Some ranges may be filtered out as degenerate (e.g., typo range), so
    the actual count may be less; the split is still deterministic on
    surviving ranges in mission-spec order.
    """
    n = len(ranges)
    n_train = int(round(train_fraction * n))
    return ranges[:n_train], ranges[n_train:]


def windows_for_ranges(ranges: list[AnomalyRange], window_size: int = 100) -> list[int]:
    """Return list of window-start indices (in stride=1 test-data window space) covered
    by the given anomaly ranges. Window of length `window_size` ending at downsampled row r
    is the prediction for row r; we treat any window where row r ∈ [s_ds, e_ds) as covered.
    """
    indices = []
    for r in ranges:
        for ds_row in range(r.start_row_ds, r.end_row_ds):
            window_start = ds_row - (window_size - 1)
            if window_start < 0:
                continue
            indices.append(window_start)
    return sorted(set(indices))


def get_swat_anomaly_ranges_ds(test_csv_path: str | None = None) -> list[AnomalyRange]:
    """Derive SWaT anomaly ranges from test.csv attack labels.

    SWaT test.csv has an `attack` column (0/1 per row). Contiguous runs of
    attack==1 form anomaly ranges. Returned ranges use downsampled-row
    coordinates (which match the test_loader's row indexing for SWaT).
    """
    import pandas as pd
    from pathlib import Path

    if test_csv_path is None:
        test_csv_path = str(Path(__file__).resolve().parent.parent / "repos/GDN/data/swat/test.csv")
    df = pd.read_csv(test_csv_path, index_col=0)
    labels = df["attack"].values.astype(int)
    n = len(labels)
    ranges = []
    in_run = False
    s = 0
    range_idx = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_run:
            in_run = True
            s = i
        elif v == 0 and in_run:
            in_run = False
            ranges.append(AnomalyRange(range_idx, s * 10, i * 10, s, i))
            range_idx += 1
    if in_run:
        ranges.append(AnomalyRange(range_idx, s * 10, n * 10, s, n))
    return ranges


def get_anomaly_ranges_ds(dataset: str, num_test_rows_ds: int | None = None) -> list[AnomalyRange]:
    """Dispatcher for dataset-specific anomaly-range extraction."""
    if dataset == "wadi":
        if num_test_rows_ds is None:
            raise ValueError("num_test_rows_ds required for WADI")
        return get_wadi_anomaly_ranges_ds(num_test_rows_ds=num_test_rows_ds)
    elif dataset == "swat":
        return get_swat_anomaly_ranges_ds()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


__all__ = [
    "AnomalyRange",
    "get_wadi_anomaly_ranges_ds",
    "get_swat_anomaly_ranges_ds",
    "get_anomaly_ranges_ds",
    "split_ranges",
    "windows_for_ranges",
]
