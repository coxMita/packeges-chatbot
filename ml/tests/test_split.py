"""The 70/15/15 split: sizes, grouping, and the future holdout."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from split import dataset_splits  # noqa: E402


def _frame(n_groups: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = int(g % 3 == 0)
        # Malware campaigns come in big families; benign packages mostly alone.
        size = int(rng.integers(1, 30)) if label else 1
        reported = "2026-01-10" if label and g % 7 == 0 else ("2024-03-01" if label else "")
        for i in range(size):
            rows.append({"package": f"p{g}-{i}", "version": "1.0", "label": label,
                         "group": f"g{g}", "reported": reported, "pool": "x"})
    return pd.DataFrame(rows)


def test_split_is_70_15_15_by_rows_per_class():
    df = _frame()
    s = dataset_splits(df)
    rest = len(df) - len(s.future)
    for part, want in ((s.train, 0.70), (s.val, 0.15), (s.test, 0.15)):
        assert abs(len(part) / rest - want) < 0.04
    for label in (0, 1):
        n = sum(int((p["label"] == label).sum()) for p in (s.train, s.val, s.test))
        assert abs((s.test["label"] == label).sum() / n - 0.15) < 0.05


def test_no_group_straddles_two_sets():
    s = dataset_splits(_frame())
    groups = [set(p["group"]) for p in (s.train, s.val, s.test, s.future)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not groups[i] & groups[j]


def test_future_holds_only_late_malware_from_unseen_families():
    s = dataset_splits(_frame(), cutoff="2025-06-01")
    assert len(s.future) and (s.future["label"] == 1).all()
    assert (s.future["reported"] >= "2025-06-01").all()


def test_split_is_deterministic():
    a, b = dataset_splits(_frame()), dataset_splits(_frame())
    assert a.test["package"].tolist() == b.test["package"].tolist()
