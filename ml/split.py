"""Shared train/test splitting.

Lives in its own module because `train_gbdt.py`, `train_embed.py` and
`evaluate.py` must all use *exactly* the same split. If they drift, the
head-to-head model comparison is meaningless.

The split is grouped on package name. Two releases of the same compromised
library are near-identical; if one lands in train and the other in test, the
model scores well by memorising the package rather than recognising the
behaviour. Grouping removes that inflation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import RANDOM_SEED

TEST_SIZE = 0.2


def grouped_split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED):
    """Split `df` into (train, test) with no package name appearing in both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["group"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(train["group"]) & set(test["group"])
    assert not overlap, f"group leakage across the split: {sorted(overlap)[:5]}"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return df[feature_cols].to_numpy(dtype=np.float32), df["label"].to_numpy(dtype=np.int32)
