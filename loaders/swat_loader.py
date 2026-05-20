"""
SWaT canonical preprocessing for GDN / TopoGDN / BETA reproduction.

H_PI_4 (Phase 0 closeout): infrastructure ready, data load BLOCKED on iTrust DUA
(chk_01KQGQDN5J2C9ESNT9BBPTE088). The canonical SWaT files are
`SWaT_dataset_Jul 19.xlsx` and `SWaT_dataset_Jul 19 v2.xlsx` from the iTrust
A4 & A5 release; the user has SWaT.zip downloaded but those specific xlsx
files are 135-byte _Error.txt placeholders due to iTrust's bulk-zip size cap.

When the canonical xlsx files arrive (via individual download from the iTrust
portal), this loader will produce the GDN-format `swat_train.csv`, `swat_test.csv`,
`list.txt` triple expected by `repos/GDN/data/swat/` and
`repos/TopoGDN/data/swat/`.

Pipeline (mirrors `repos/GDN/scripts/process_swat.py` exactly):
  1. Read `SWaT_Dataset_Normal_v1.csv` + `SWaT_Dataset_Attack_v0.csv`
     OR `SWaT_dataset_Jul 19 v2.xlsx` (the rebranded A1+A2 = A4+A5 release).
  2. Drop first column (Timestamp); fillna(mean) then fillna(0).
  3. Split off 'attack' column as labels (1=attack, 0=normal).
  4. Min-Max [0,1] normalization fit on train; applied to both.
  5. Downsample-by-10 with median (1Hz → 0.1Hz, ~each row = 10s).
  6. Trim first 2160 downsampled train rows (= 6 hours warmup at original 1Hz).
  7. Write to `data/swat/{train,test,list}.csv|txt` per GDN convention.

BETA Table 2 expected post-preprocessing counts:
  - 51 sensors
  - 47515 train rows / 44986 test rows
  - 11.97% test anomaly rate
  - This matches the canonical SWaT.A1 (Dec 2015) / A2 (Jul 2017) release.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data/swat/SWaT_Dataset_Normal_v1.csv"
DEFAULT_ATTACK_CSV = PROJECT_ROOT / "data/swat/SWaT_Dataset_Attack_v0.csv"
DEFAULT_TRAIN_XLSX = PROJECT_ROOT / "data/swat/SWaT_dataset_Jul 19 v2.xlsx"

EXPECTED_FEATURE_COUNT = 51
EXPECTED_TRAIN_ROWS = 47515
EXPECTED_TEST_ROWS = 44986


def downsample(data: np.ndarray, labels: np.ndarray, factor: int = 10):
    """Median-downsample features; max-aggregate labels (any-anomaly-in-window -> 1)."""
    orig_len, col_num = data.shape
    down_len = orig_len // factor
    trimmed = data[: down_len * factor].T.reshape(col_num, -1, factor)
    d_data = np.median(trimmed, axis=2).T
    d_labels = np.round(np.max(labels[: down_len * factor].reshape(-1, factor), axis=1))
    return d_data, d_labels


def load_swat_canonical(train_path: Path, attack_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and clean canonical SWaT (A1/A2 or repackaged A4/A5).

    Both files expected to have:
      - First column: Timestamp (drop)
      - Last column: 'attack' or 'Normal/Attack' (binary anomaly label)
      - 51 sensor feature columns in between

    Sensor labels in the official xlsx use 'Normal'/'Attack' strings rather than 0/1.
    We convert: 'Normal' or 'A ttack' (with internal space typo present in original) → 0;
    everything else → 1.
    """
    if not train_path.exists() or not attack_path.exists():
        raise FileNotFoundError(
            f"SWaT canonical files not found.\n"
            f"  train: {train_path} (exists={train_path.exists()})\n"
            f"  attack: {attack_path} (exists={attack_path.exists()})\n"
            f"BLOCKED on iTrust DUA per chk_01KQGQDN5J2C9ESNT9BBPTE088."
        )

    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(attack_path, index_col=0)

    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()

    # Locate the label column. Standard release uses 'Normal/Attack'.
    label_col_candidates_train = [c for c in train.columns if c in {"Normal/Attack", "attack", "Label"}]
    label_col_candidates_test = [c for c in test.columns if c in {"Normal/Attack", "attack", "Label"}]
    if not label_col_candidates_test:
        raise ValueError(f"No label column found in test. Last 5 cols: {list(test.columns)[-5:]}")
    label_col = label_col_candidates_test[0]

    # Convert string labels to 0/1.
    def _label_to_int(v):
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip().lower()
        return 0 if s.startswith("normal") else 1

    test_labels = test[label_col].apply(_label_to_int).values.astype(np.int64)
    test = test.drop(columns=[label_col])
    if label_col_candidates_train:
        train = train.drop(columns=label_col_candidates_train)

    train = train.fillna(train.mean(numeric_only=True)).fillna(0)
    test = test.fillna(test.mean(numeric_only=True)).fillna(0)

    if train.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Train has {train.shape[1]} features, expected {EXPECTED_FEATURE_COUNT}.")
    if test.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Test has {test.shape[1]} features, expected {EXPECTED_FEATURE_COUNT}.")

    test = test[train.columns]  # align column order
    train_labels = np.zeros(len(train), dtype=np.int64)
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
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--attack", default=str(DEFAULT_ATTACK_CSV))
    parser.add_argument(
        "--out-dirs",
        nargs="+",
        default=[
            str(PROJECT_ROOT / "repos/GDN/data/swat"),
            str(PROJECT_ROOT / "repos/TopoGDN/data/swat"),
        ],
    )
    args = parser.parse_args()

    train, test, train_labels, test_labels = load_swat_canonical(Path(args.train), Path(args.attack))
    print(f"[load] train shape={train.shape} test shape={test.shape}")
    print(f"[load] test attack rate (raw): {test_labels.mean():.4%}")

    train_df, test_df, feature_cols = preprocess_for_gdn(train, test, train_labels, test_labels)
    print(f"[preprocess] post-downsample+trim train shape={train_df.shape}")
    print(f"[preprocess] post-downsample test shape={test_df.shape}")
    print(f"[preprocess] post-downsample test attack rate: {test_df['attack'].mean():.4%}")
    print(f"[preprocess] BETA Table 2 expects: {EXPECTED_TRAIN_ROWS} train, {EXPECTED_TEST_ROWS} test, 11.97% anomalies")

    for out in args.out_dirs:
        out_dir = Path(out)
        write_outputs(out_dir, train_df, test_df, feature_cols)
        print(f"[write] {out_dir}/  -> train.csv ({len(train_df)} rows), test.csv ({len(test_df)} rows), list.txt ({len(feature_cols)} features)")


if __name__ == "__main__":
    main()
