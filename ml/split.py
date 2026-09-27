"""Shared train / validation / test splitting.

Lives in its own module because `train_gbdt.py`, `train_embed.py`,
`calibrate.py` and `evaluate.py` must all use *exactly* the same split. If they
drift, the head-to-head model comparison is meaningless.

The scheme is 70 / 15 / 15, plus a fourth set outside it:

    train    70%  fits the models (thresholds from grouped CV inside it)
    val      15%  fits the calibrator and the similarity flag -- model selection
    test     15%  touched once, for the numbers in the report
    future   --   malware first reported on or after FUTURE_CUTOFF whose
                  behaviour fingerprint never appears earlier. Kept out of all
                  three sets, so it stands in for "malware nobody trained on".

The split is grouped on package name, merged with every other package that
has the exact same feature vector. Two releases of the same compromised
library are near-identical, and a campaign uploads one payload under hundreds
of names; if copies land on both sides, the model scores well by memorising
the sample rather than recognising the behaviour. Grouping removes that
inflation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import RANDOM_SEED

TEST_SIZE = 0.15
VAL_SIZE = 0.15
FUTURE_CUTOFF = "2025-06-01"

# Depend on the name rather than the code, so they would split copies apart.
NAME_DEPENDENT = {"pkg_typosquat_distance"}


def merge_duplicate_groups(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Union name groups that share an identical behaviour fingerprint.

    Returns a new group label per row: the smallest name in its merged set.
    """
    cols = [c for c in feature_cols if c not in NAME_DEPENDENT]
    fingerprint = pd.util.hash_pandas_object(df[cols].round(4), index=False)

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for name, fp in zip(df["group"], fingerprint):
        a, b = find(name), find(f"fp:{fp}")
        if a != b:
            parent[b] = a

    # Label each merged set by its alphabetically first package name.
    label: dict[str, str] = {}
    for name in df["group"]:
        root = find(name)
        label[root] = min(label.get(root, name), name)
    return df["group"].map(lambda n: label[find(n)])


def grouped_split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED):
    """Split `df` into (train, test) with no package name appearing in both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["group"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(train["group"]) & set(test["group"])
    assert not overlap, f"group leakage across the split: {sorted(overlap)[:5]}"
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_future(df: pd.DataFrame, cutoff: str = FUTURE_CUTOFF
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rest, future): future = malware reported >= cutoff with no older copy."""
    dated = df["reported"].fillna("") != ""
    late = (df["label"] == 1) & dated & (df["reported"] >= cutoff)
    old_groups = set(df.loc[~late, "group"])
    future = late & ~df["group"].isin(old_groups)
    return df[~future].reset_index(drop=True), df[future].reset_index(drop=True)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    future: pd.DataFrame

    def name_of(self) -> dict[str, str]:
        """`package@version` -> split name, for tables that hold a subset."""
        out = {}
        for name in ("train", "val", "test", "future"):
            d = getattr(self, name)
            out.update({f"{p}@{v}": name for p, v in zip(d["package"], d["version"])})
        return out


def dataset_splits(df: pd.DataFrame, seed: int = RANDOM_SEED,
                   cutoff: str = FUTURE_CUTOFF) -> Splits:
    """The 70 / 15 / 15 grouped split, with the future malware held out first.

    Fractions are of *rows*, per class. GroupShuffleSplit counts groups instead,
    and one malware campaign can be a single group of hundreds of packages, so
    it lands nowhere near 70/15/15. Here groups are shuffled within each class
    (by the group's majority label) and dealt out until each set holds its share
    of that class's rows.
    """
    rest, future = split_future(df, cutoff)
    rng = np.random.default_rng(seed)
    sizes = rest.groupby("group").size()
    majority = rest.groupby("group")["label"].mean().round().astype(int)

    where: dict[str, str] = {}
    for label in (0, 1):
        groups = sizes[majority == label].index.to_numpy()
        groups = groups[rng.permutation(len(groups))]
        total, seen = int(sizes[groups].sum()), 0
        for g in groups:
            frac = seen / max(total, 1)
            where[g] = ("test" if frac < TEST_SIZE
                        else "val" if frac < TEST_SIZE + VAL_SIZE else "train")
            seen += int(sizes[g])

    part = rest["group"].map(where)
    pick = lambda name: rest[part == name].reset_index(drop=True)  # noqa: E731
    return Splits(pick("train"), pick("val"), pick("test"), future)


def xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return df[feature_cols].to_numpy(dtype=np.float32), df["label"].to_numpy(dtype=np.int32)


def with_synthetic(train: pd.DataFrame, path) -> pd.DataFrame:
    """Append the synthetic trojanized packages (ml/augment.py) that are safe to
    train on: both the host and the donor code come from this training split.
    Synthetic rows are never part of a test set."""
    from pathlib import Path
    if not Path(path).exists():
        return train
    aug = pd.read_parquet(path)
    groups = set(train["group"])
    ok = aug["host_group"].isin(groups) & aug["donor_group"].isin(groups)
    return pd.concat([train, aug[ok].drop(columns=["host_group", "donor_group"])],
                     ignore_index=True)
