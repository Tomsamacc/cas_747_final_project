import argparse
import logging
import os
import sys

from src.data_processing.features import normalize_features_l2
from src.data_processing.io import save_processed
from src.data_processing.load_data import load_graph, sync_ds_old_env_with_config
from src.data_processing.splits import build_lsgnn_split_stack, stack_ogbn_arxiv_official_masks
from src.utils.log_setup import get_logger, setup_logging


def build_processed_payload(
    dataset_name,
    raw_root=None,
    undirected=False,
    num_splits=10,
    seed_base=28,
    train_r=0.48,
    val_r=0.32,
    test_r=0.20,
):
    name = dataset_name.lower().replace("_", "-")
    if name in ("ogbn-arxiv", "arxiv-year"):
        undirected = True

    sync_ds_old_env_with_config(dataset_name)
    data = load_graph(dataset_name, raw_root=raw_root, undirected=undirected)
    y = data.y.squeeze()
    x = normalize_features_l2(data.x)

    if name == "ogbn-arxiv":
        split_idx = getattr(data, "_ogb_split_idx")
        train_m, val_m, test_m = stack_ogbn_arxiv_official_masks(
            data.num_nodes, split_idx, num_splits=num_splits
        )
    else:
        tr, va, te = train_r, val_r, test_r
        if name == "arxiv-year":
            tr, va, te = 0.5, 0.25, 0.25
        train_m, val_m, test_m = build_lsgnn_split_stack(
            data,
            dataset_name,
            num_splits=num_splits,
            global_seed_base=seed_base,
            train_r=tr,
            val_r=va,
            test_r=te,
        )

    num_classes = int(y.unique().shape[0])

    return {
        "x": x,
        "y": y,
        "edge_index": data.edge_index,
        "train_mask": train_m,
        "val_mask": val_m,
        "test_mask": test_m,
        "num_classes": num_classes,
        "meta": {
            "dataset": dataset_name.lower(),
            "num_splits": train_m.size(1),
            "undirected": undirected,
            "train_val_test": (train_r, val_r, test_r),
            "seed_base": seed_base,
        },
    }


def preprocess_dataset(
    name,
    raw_root=None,
    undirected=False,
    num_splits=10,
    seed_base=28,
    train_r=0.48,
    val_r=0.32,
    test_r=0.20,
):
    payload = build_processed_payload(
        name,
        raw_root=raw_root,
        undirected=undirected,
        num_splits=num_splits,
        seed_base=seed_base,
        train_r=train_r,
        val_r=val_r,
        test_r=test_r,
    )
    return save_processed(name.lower(), payload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--raw-root", type=str, default=None)
    p.add_argument("--undirected", action="store_true")
    p.add_argument("--num-splits", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=28)
    args = p.parse_args()

    setup_logging(file_name="preprocess.log", subdir=args.dataset)
    log = get_logger("preprocess")

    log.info(
        "start dataset=%s num_splits=%s undirected=%s",
        args.dataset,
        args.num_splits,
        args.undirected,
    )

    path = preprocess_dataset(
        args.dataset,
        raw_root=args.raw_root,
        undirected=args.undirected,
        num_splits=args.num_splits,
        seed_base=args.seed_base,
    )
    log.info("saved %s", path)
    sys.stdout.flush()
    sys.stderr.flush()
    for h in logging.getLogger("lsgnn").handlers:
        try:
            h.flush()
        except Exception:
            pass

    if os.environ.get("LSGNN_PREPROCESS_NORMAL_EXIT", "").lower() not in ("1", "true", "yes"):
        os._exit(0)


if __name__ == "__main__":
    main()
