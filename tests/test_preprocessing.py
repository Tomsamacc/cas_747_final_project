import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.data import Data

from src.data_processing.features import normalize_features_l2
from src.data_processing.io import load_processed, save_processed, with_split
from src.data_processing.preprocess import preprocess_dataset
from src.data_processing.splits import build_lsgnn_split_stack, class_balanced_split


def check_normalize_features_l2_rows_unit():
    x = torch.randn(20, 8)
    z = normalize_features_l2(x)
    norms = z.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def check_with_split_1d_masks_unchanged():
    n = 15
    payload = {
        "x": torch.zeros(n, 3),
        "y": torch.zeros(n, dtype=torch.long),
        "edge_index": torch.zeros(2, 0, dtype=torch.long),
        "train_mask": torch.ones(n, dtype=torch.bool),
        "val_mask": torch.zeros(n, dtype=torch.bool),
        "test_mask": torch.zeros(n, dtype=torch.bool),
        "num_classes": 2,
    }
    out = with_split(payload, split_idx=3)
    assert out["train_mask"].shape == (n,)
    assert out["train_mask"].equal(payload["train_mask"])


def check_with_split_2d_selects_column():
    n = 10
    tm = torch.zeros(n, 4, dtype=torch.bool)
    tm[:, 2] = True
    payload = {
        "x": torch.zeros(n, 3),
        "y": torch.zeros(n, dtype=torch.long),
        "edge_index": torch.zeros(2, 0, dtype=torch.long),
        "train_mask": tm,
        "val_mask": tm.clone(),
        "test_mask": tm.clone(),
        "num_classes": 2,
    }
    out = with_split(payload, split_idx=2)
    assert out["train_mask"].shape == (n,)
    assert out["train_mask"].all()


def check_class_balanced_split_partition():
    n = 100
    y = torch.randint(0, 5, (n,))
    tr, va, te = class_balanced_split(y, n, 5, seed=0)
    assert (tr & va).sum() == 0 and (tr & te).sum() == 0 and (va & te).sum() == 0
    assert tr.sum() + va.sum() + te.sum() == n


def check_build_lsgnn_split_stack_hetero_columns():
    n = 12
    data = Data(
        y=torch.zeros(n, dtype=torch.long),
        train_mask=torch.ones(n, 3, dtype=torch.bool),
        val_mask=torch.zeros(n, 3, dtype=torch.bool),
        test_mask=torch.zeros(n, 3, dtype=torch.bool),
    )
    tr, va, te = build_lsgnn_split_stack(data, "cornell", num_splits=2)
    assert tr.shape == (n, 2)


def check_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        proc = os.path.join(tmp, "processed")
        os.makedirs(proc)
        with patch("src.data_processing.paths.PROCESSED_DIR", proc), patch(
            "src.data_processing.io.PROCESSED_DIR", proc
        ):
            payload = {
                "x": torch.ones(2, 1),
                "y": torch.zeros(2, dtype=torch.long),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "train_mask": torch.tensor([True, False]),
                "val_mask": torch.tensor([False, True]),
                "test_mask": torch.tensor([False, False]),
                "num_classes": 1,
            }
            save_processed("tmp_ds", payload)
            loaded = load_processed("tmp_ds")
            assert loaded["x"].equal(payload["x"])


def _fake_load_graph_for_cora(name, raw_root=None, undirected=False):
    n = 32
    return Data(
        x=torch.randn(n, 7),
        y=torch.randint(0, 4, (n,)),
        edge_index=torch.randint(0, n, (2, 64)),
    )


def check_preprocess_dataset_runs_in_isolated_dir():
    with tempfile.TemporaryDirectory() as tmp:
        proc = os.path.join(tmp, "processed")
        raw = os.path.join(tmp, "raw")
        os.makedirs(proc)
        os.makedirs(raw)
        with patch("src.data_processing.paths.PROCESSED_DIR", proc), patch(
            "src.data_processing.paths.RAW_DIR", raw
        ), patch("src.data_processing.io.PROCESSED_DIR", proc), patch(
            "src.data_processing.preprocess.load_graph",
            _fake_load_graph_for_cora,
        ):
            out_path = preprocess_dataset("cora", num_splits=3, seed_base=28)
            assert os.path.isfile(out_path)
            assert str(out_path).startswith(proc)

            loaded = load_processed("cora")
            assert loaded["x"].shape[0] == 32
            assert loaded["train_mask"].shape == (32, 3)


def main():
    check_normalize_features_l2_rows_unit()
    print("ok: normalize_features_l2")
    check_with_split_1d_masks_unchanged()
    print("ok: with_split 1d")
    check_with_split_2d_selects_column()
    print("ok: with_split 2d")
    check_class_balanced_split_partition()
    print("ok: class_balanced_split")
    check_build_lsgnn_split_stack_hetero_columns()
    print("ok: build_lsgnn_split_stack hetero")
    check_save_load_roundtrip()
    print("ok: save/load roundtrip (temp dir)")
    check_preprocess_dataset_runs_in_isolated_dir()
    print("ok: preprocess_dataset (temp dir, fake graph)")


if __name__ == "__main__":
    main()
