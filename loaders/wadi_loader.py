"""
WADI A1 (Oct 2017) canonical preprocessing for GDN / TopoGDN / BETA reproduction.

BETA Table 2 reports 118795 train / 17275 test rows for WADI. WADI A1 has 1,209,601 raw
training rows at 1 Hz; downsample-by-10 yields 120,960; trimming the 2160-row warmup
yields 118,800 — matching BETA within 5 rows. WADI A2 (the _new reissue) only has
~785K raw training rows and produces ~76K post-preprocessing — does NOT match BETA's
counts. Conclusion: BETA uses A1, not A2.

Pipeline (mirrors repos/GDN/scripts/process_wadi.py + repos/TopoGDN/scripts/process_wadi.py
+ repos/TopoGDN/scripts/wadi_mark_label.py):
  1. Read WADI_14days.csv (train; A1, 4-line preamble, header on row 5).
     Read WADI_attackdata.csv (test; A1, header on row 1, NO label column).
  2. Drop Date + Time metadata cols (Row is consumed by index_col=0), leaving 127 sensors.
  3. Strip the 46-char `\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\` prefix from
     A1 column names so they read as `1_AIT_001_PV`, etc.
  4. Generate test labels from the 16 known WADI attack time-windows (TopoGDN convention,
     base date 10/9/2017 18:00:00 = row 1 of the attack CSV).
  5. fillna(mean) then fillna(0).
  6. Min-Max [0,1] fit on train, applied to both.
  7. Downsample-by-10 with median; labels by max-aggregation.
  8. Trim first 2160 downsampled rows of train (~6h warmup at original 1Hz).
  9. Write train.csv, test.csv, list.txt to each detector's data/wadi/ folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = PROJECT_ROOT / "data/WaDi/WaDi/WADI.A1_9 Oct 2017/WADI_14days.csv"
DEFAULT_TEST = PROJECT_ROOT / "data/WaDi/WaDi/WADI.A1_9 Oct 2017/WADI_attackdata.csv"

EXPECTED_FEATURE_COUNT = 127
EXPECTED_TRAIN_PREAMBLE_LINES = 4  # A1 train CSV preamble (Created, Number of rows, Interpolation interval, blank)
COLUMN_PREFIX_LEN = 46  # `\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\`

# Attack time-ranges from repos/TopoGDN/scripts/wadi_mark_label.py.
# Base of WADI attack CSV: 10/9/2017 18:00:00 == row 1.
WADI_ATTACK_BASE = datetime(2017, 10, 9, 18, 0, 0)
WADI_ATTACK_RANGES = [
    ("10/9/2017", "19:25:00", "10/9/2017", "19:50:16"),
    ("10/10/2017", "10:24:10", "10/10/2017", "10:34:00"),
    ("10/10/2017", "10:55:00", "10/10/2017", "11:24:00"),
    ("10/10/2017", "11:07:46", "10/10/2017", "11:12:15"),
    ("10/10/2017", "11:30:40", "10/10/2017", "11:44:50"),
    ("10/10/2017", "13:39:30", "10/10/2017", "13:50:40"),
    ("10/10/2017", "14:48:17", "10/10/2017", "14:59:55"),
    ("10/10/2017", "14:53:44", "10/10/2017", "15:00:32"),
    ("10/10/2017", "17:40:00", "10/10/2017", "17:49:40"),
    # Note: TopoGDN's script has start_date=10/11 but end_date=10/10 — preserved verbatim
    # so reproduction matches their label set; the inverted range produces no labeled rows.
    ("10/11/2017", "10:55:00", "10/10/2017", "10:56:27"),
    ("10/11/2017", "11:17:54", "10/11/2017", "11:31:20"),
    ("10/11/2017", "11:36:31", "10/11/2017", "11:47:00"),
    ("10/11/2017", "11:59:00", "10/11/2017", "12:05:00"),
    ("10/11/2017", "12:07:30", "10/11/2017", "12:10:52"),
    ("10/11/2017", "12:16:00", "10/11/2017", "12:25:36"),
    ("10/11/2017", "15:26:30", "10/11/2017", "15:37:00"),
]


def _parse(date_str: str, time_str: str) -> datetime:
    return datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S")


def derive_attack_labels(num_rows: int, base: datetime = WADI_ATTACK_BASE) -> np.ndarray:
    labels = np.zeros(num_rows, dtype=np.int64)
    for start_d, start_t, end_d, end_t in WADI_ATTACK_RANGES:
        start_sec = int((_parse(start_d, start_t) - base).total_seconds())
        end_sec = int((_parse(end_d, end_t) - base).total_seconds())
        # Row 1 of the CSV maps to base second; convert to 0-based python indices.
        start_idx = max(start_sec, 0)
        end_idx = min(end_sec + 1, num_rows)  # inclusive end -> exclusive slice
        if start_idx < end_idx:
            labels[start_idx:end_idx] = 1
    return labels


def downsample(data: np.ndarray, labels: np.ndarray, factor: int = 10):
    orig_len, col_num = data.shape
    down_len = orig_len // factor
    trimmed = data[: down_len * factor].T.reshape(col_num, -1, factor)
    d_data = np.median(trimmed, axis=2).T
    d_labels = np.round(np.max(labels[: down_len * factor].reshape(-1, factor), axis=1))
    return d_data, d_labels


def load_wadi_a1(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    if not train_path.exists():
        raise FileNotFoundError(f"WADI A1 train CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"WADI A1 test CSV not found: {test_path}")

    # Train: 4-line preamble, header on row 5 (skiprows=4 skips lines 0..3).
    train = pd.read_csv(train_path, skiprows=EXPECTED_TRAIN_PREAMBLE_LINES, index_col=0)
    # Test: header on row 1, no preamble, no label column.
    test = pd.read_csv(test_path, index_col=0)

    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    # Drop Date + Time metadata columns.
    train = train.drop(columns=["Date", "Time"])
    test = test.drop(columns=["Date", "Time"])

    # Strip 46-char `\\WIN-...\LOG_DATA\` prefix from A1 column names (both files).
    def _strip_prefix(cols):
        return [c[COLUMN_PREFIX_LEN:] if len(c) > COLUMN_PREFIX_LEN else c for c in cols]

    train.columns = _strip_prefix(train.columns)
    test.columns = _strip_prefix(test.columns)

    if train.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Train has {train.shape[1]} features, expected {EXPECTED_FEATURE_COUNT}.")
    if test.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Test has {test.shape[1]} features, expected {EXPECTED_FEATURE_COUNT}.")

    # Align test column order to train.
    test = test[train.columns]

    # NaN handling: per-column mean, fall back to 0.
    train = train.fillna(train.mean(numeric_only=True)).fillna(0)
    test = test.fillna(test.mean(numeric_only=True)).fillna(0)

    train_labels = np.zeros(len(train), dtype=np.int64)
    test_labels = derive_attack_labels(len(test))
    return train, test, train_labels, test_labels


def preprocess_for_gdn(train: pd.DataFrame, test: pd.DataFrame, train_labels: np.ndarray, test_labels: np.ndarray):
    feature_cols = list(train.columns)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(train.values)
    x_train = scaler.transform(train.values)
    x_test = scaler.transform(test.values)

    d_train_x, d_train_y = downsample(x_train, train_labels, 10)
    d_test_x, d_test_y = downsample(x_test, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns=feature_cols)
    test_df = pd.DataFrame(d_test_x, columns=feature_cols)
    train_df["attack"] = d_train_y.astype(int)
    test_df["attack"] = d_test_y.astype(int)

    train_df = train_df.iloc[2160:].reset_index(drop=True)
    return train_df, test_df, feature_cols


def write_outputs(out_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv")
    test_df.to_csv(out_dir / "test.csv")
    (out_dir / "list.txt").write_text("\n".join(feature_cols) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument(
        "--out-dirs",
        nargs="+",
        default=[
            str(PROJECT_ROOT / "repos/GDN/data/wadi"),
            str(PROJECT_ROOT / "repos/TopoGDN/data/wadi"),
        ],
    )
    args = parser.parse_args()

    train, test, train_labels, test_labels = load_wadi_a1(Path(args.train), Path(args.test))
    print(f"[load] train shape={train.shape} test shape={test.shape}")
    print(f"[load] test attack rate (raw): {test_labels.mean():.4%}  ({test_labels.sum()} of {len(test_labels)})")

    train_df, test_df, feature_cols = preprocess_for_gdn(train, test, train_labels, test_labels)
    print(f"[preprocess] post-downsample+trim train shape={train_df.shape}")
    print(f"[preprocess] post-downsample test shape={test_df.shape}")
    print(f"[preprocess] post-downsample test attack rate: {test_df['attack'].mean():.4%}")
    print(f"[preprocess] features: {len(feature_cols)} (expected {EXPECTED_FEATURE_COUNT})")
    print(f"[preprocess] BETA Table 2 expects: 118795 train, 17275 test, 5.99% anomalies")

    for out in args.out_dirs:
        out_dir = Path(out)
        write_outputs(out_dir, train_df, test_df, feature_cols)
        print(f"[write] {out_dir}/  -> train.csv ({len(train_df)} rows), test.csv ({len(test_df)} rows), list.txt ({len(feature_cols)} features)")


if __name__ == "__main__":
    main()
