import argparse
import os

import torch

from src.data_processing.io import load_processed, with_split
from src.models.model import LSGNN
from src.utils.helpers import accuracy, build_l_h_filters, get_device
from src.utils.log_setup import get_logger, setup_logging


def load_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint at {path} missing 'state_dict' key.")
    return ckpt


def evaluate_checkpoint(
    ckpt_path, processed_name, split_idx_override=None, return_outputs=False
):
    device = get_device()
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {})
    ds = ckpt.get("dataset", "unknown")
    split_idx = (
        int(split_idx_override)
        if split_idx_override is not None
        else int(cfg.get("split_idx", 0))
    )

    data = with_split(load_processed(processed_name), split_idx)
    x = data["x"].to(device)
    y = data["y"].to(device)
    edge_index = data["edge_index"].to(device)
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)
    num_classes = int(data["num_classes"])
    num_nodes = x.size(0)

    model = LSGNN(
        in_channels=x.size(1),
        out_channels=num_classes,
        num_nodes=num_nodes,
        hidden_channels=cfg.get("hidden_channels", 16),
        K=cfg.get("K", 5),
        beta=cfg.get("beta", 1.0),
        gamma=cfg.get("gamma", 0.5),
        dropout=cfg.get("dropout", 0.5),
        method=cfg.get("method", "norm2"),
        num_reduce_layers=cfg.get("num_reduce_layers", 1),
        use_A_embedding=cfg.get("use_A_embedding", False),
        out_norm=cfg.get("out_norm", True),
        out_mlp=cfg.get("out_mlp", False),
        use_irdc=cfg.get("use_irdc", True),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    filters = build_l_h_filters(
        edge_index=edge_index,
        num_nodes=num_nodes,
        beta=cfg.get("beta", 1.0),
        transposed=cfg.get("transposed", True),
    )
    with torch.no_grad():
        dist, x_out = model.precompute_dist_and_prop(x, edge_index, filters)
        logits = model(x, edge_index, dist, x_out)

    train_acc = accuracy(logits, y, train_mask)
    val_acc = accuracy(logits, y, val_mask)
    test_acc = accuracy(logits, y, test_mask)

    preds = logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)
    conf = probs.max(dim=1).values

    log = get_logger("evaluate")
    log.info(
        "checkpoint=%s dataset=%s train_acc=%.4f val_acc=%.4f test_acc=%.4f",
        os.path.basename(ckpt_path),
        ds,
        train_acc,
        val_acc,
        test_acc,
    )
    metrics = {
        "dataset": ds,
        "split_idx": split_idx,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }
    if not return_outputs:
        return metrics
    metrics.update(
        {
            "logits": logits.detach().cpu(),
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(),
            "conf": conf.detach().cpu(),
            "y": y.detach().cpu(),
            "train_mask": train_mask.detach().cpu(),
            "val_mask": val_mask.detach().cpu(),
            "test_mask": test_mask.detach().cpu(),
            "num_classes": num_classes,
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--processed-name", type=str, default="cora")
    parser.add_argument("--split-idx", type=int, default=None)
    args = parser.parse_args()

    setup_logging(file_name="evaluate.log", subdir=args.processed_name)
    metrics = evaluate_checkpoint(
        args.ckpt,
        args.processed_name,
        split_idx_override=args.split_idx,
    )
    print(metrics)


if __name__ == "__main__":
    main()
