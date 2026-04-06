import argparse
import json
import os
from copy import deepcopy
from time import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_processing.io import load_processed, with_split
from src.data_processing.load_data import sync_ds_old_env_with_config
from src.models.model import LSGNN
from src.utils.helpers import accuracy, build_l_h_filters, get_device, plot_training_curves, set_seed
from src.utils.log_setup import get_logger, setup_logging

# Mirrors LSGNN-master/v2/config.py (default_config + lsgnn).
LARGE_GRAPH = frozenset({"ogbn-arxiv", "arxiv-year"})

CONFIG_DATASETS = frozenset(
    {
        "actor",
        "arxiv-year",
        "chameleon",
        "citeseer",
        "cornell",
        "cora",
        "ogbn-arxiv",
        "pubmed",
        "squirrel",
        "texas",
        "wisconsin",
    }
)


def _normalize_dataset_name(ds: str) -> str:
    s = ds.lower().replace("_", "-")
    if s == "arxiv_year":
        s = "arxiv-year"
    return s


def load_lsgnn_config_json(ds: str = "cora") -> dict:
    name = _normalize_dataset_name(ds)
    # check if the dataset is supported
    assert name in CONFIG_DATASETS, (
        f"dataset={ds!r} -> {name!r} not in CONFIG_DATASETS; "
        f"expected one of {sorted(CONFIG_DATASETS)}"
    )
    #get the path to the config file
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    config_path = os.path.join(parent, "configs", f"{name}.json")
    # check if the config file exists
    assert os.path.isfile(config_path), f"missing config file: {config_path}"
    # load the config file
    with open(config_path, encoding="utf-8") as file:
        return json.load(file)

# If previous config file is not found, use the default config
def default_lsgnn_config(ds="cora"):
    ds = ds.lower().replace("_", "-")
    if ds == "arxiv_year":
        ds = "arxiv-year"

    cfg = {
        "num_epochs": 200,
        "train_val_test": [0.48, 0.32, 0.20],
        "undirected": False,
        "transposed": True,
        "hidden_channels": 16,
        "lr": 0.01,
        "wd": 5e-4,
        "dropout": 0.5,
        "adamw": False,
        "K": 5,
        "beta": 1.0,
        "gamma": 0.5,
        "method": "norm2",
        "num_reduce_layers": 1,
        "use_A_embedding": False,
        "out_norm": True,
        "out_mlp": False,
        "use_irdc": True,
        "split_idx": 0,
        "ds_old": False,
        "use_tqdm": True,
        "save_by_epoch": False,
    }

    if ds == "ogbn-arxiv":
        cfg["num_epochs"] = 500
        cfg["train_val_test"] = None
        cfg["undirected"] = True
        cfg["transposed"] = False
    elif ds in LARGE_GRAPH:
        cfg["train_val_test"] = [0.50, 0.25, 0.25]

    if ds == "ogbn-arxiv":
        cfg["hidden_channels"] = 64
        cfg["out_norm"] = False
        cfg["use_irdc"] = False
        cfg["wd"] = 0.001
        cfg["adamw"] = True
    elif ds in LARGE_GRAPH:
        cfg["hidden_channels"] = 32
        cfg["num_reduce_layers"] = 2
        cfg["use_A_embedding"] = True
        cfg["out_norm"] = False
        cfg["out_mlp"] = True
    elif ds == "cornell":
        cfg["ds_old"] = True

    return cfg


def train_single_run(ds, cfg, processed_name, log, model_dir="results/model_outputs", plots_dir="results/plots"):
    if log is None:
        log = get_logger("train")

    device = get_device()
    set_seed(28)

    split_idx = int(cfg.get("split_idx", 0))
    split_suffix = f"_split{split_idx}"
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
        hidden_channels=cfg["hidden_channels"],
        K=cfg["K"],
        beta=cfg["beta"],
        gamma=cfg["gamma"],
        dropout=cfg["dropout"],
        method=cfg["method"],
        num_reduce_layers=cfg["num_reduce_layers"],
        use_A_embedding=cfg["use_A_embedding"],
        out_norm=cfg["out_norm"],
        out_mlp=cfg["out_mlp"],
        use_irdc=cfg["use_irdc"],
    ).to(device)

    filters = build_l_h_filters(
        edge_index=edge_index,
        num_nodes=num_nodes,
        beta=cfg["beta"],
        transposed=cfg["transposed"],
    )
    # dist + IRDC stack fixed for all epochs
    t_precompute = time()
    with torch.no_grad():
        dist, x_out = model.precompute_dist_and_prop(x, edge_index, filters)
    precompute_wall_time_sec = time() - t_precompute

    wd = cfg.get("wd", cfg.get("weight_decay", 5e-4))
    opt_cls = torch.optim.AdamW if cfg.get("adamw", False) else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=cfg["lr"], weight_decay=wd)

    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    test_acc_list = []
    # record the best val loss
    best_val_loss = float("inf")
    
    # record the best val acc on the best val loss
    best_val_acc_on_best_loss = 0.0
    
    # record the best val loss on the best val acc
    
    best_val_loss_on_best_val_acc = float("inf")
    
    # record the best val acc
    best_val_acc = 0.0
    
    # record the best test accs
    best_test_acc_on_best_val_acc = 0.0
    best_test_acc_on_best_loss = 0.0
    
    best_loss_state = None
    best_acc_state = None

    epochs = range(cfg["num_epochs"])
    if cfg.get("use_tqdm", True):
        epochs = tqdm(epochs, ncols=70, desc=f"train {ds}{split_suffix}")

    os.makedirs(os.path.join(model_dir, ds), exist_ok=True)
    
    log.info(f"start training {ds} with config: {cfg}")

    t_epochs = time()
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, dist, x_out)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())

        with torch.no_grad():
            model.eval()
            logits = model(x, edge_index, dist, x_out)
            val_loss = F.cross_entropy(logits[val_mask], y[val_mask])
            val_acc = accuracy(logits, y, val_mask)
            test_acc = accuracy(logits, y, test_mask)
            val_loss_list.append(val_loss.item())
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            log.info(f"epoch {epoch} train_loss={loss.item():.4f} val_loss={val_loss.item():.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")
            
        if cfg.get("save_by_epoch", False) and epoch % (cfg["num_epochs"] // 20) == 0:
            ckpt_path = os.path.join(
                model_dir, ds, f"lsgnn_{ds}{split_suffix}_epoch_{epoch}.ckpt"
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "dataset": ds,
                    "metrics": {
                        "val_loss": val_loss.item(),
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                    },
                },
                ckpt_path,
            )
            log.info(f"model {epoch} saved to {ckpt_path}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_val_acc_on_best_loss = val_acc
            best_test_acc_on_best_loss = test_acc
            best_loss_state = deepcopy(model.state_dict())
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss_on_best_val_acc = val_loss.item()
            best_test_acc_on_best_val_acc = test_acc
            best_acc_state = deepcopy(model.state_dict())

    training_cost_time_sec = time() - t_epochs

    epochs_x = range(len(train_loss_list))
    curves_path = os.path.join(plots_dir, ds, f"lsgnn_{ds}{split_suffix}_curves.png")
    plot_training_curves(
        epochs_x,
        train_loss_list,
        val_loss_list,
        val_acc_list,
        test_acc_list,
        curves_path,
        suptitle=f"{ds}{split_suffix}",
    )
    log.info("training curves saved to %s", curves_path)

    ckpt_best_loss_path = os.path.join(
        model_dir, ds, f"lsgnn_{ds}{split_suffix}_best_loss.ckpt"
    )
    ckpt_best_acc_path = os.path.join(
        model_dir, ds, f"lsgnn_{ds}{split_suffix}_best_acc.ckpt"
    )
    
    torch.save(
        {
            "state_dict": best_loss_state,
            "config": cfg,
            "dataset": ds,
            "metrics": {
                "val_acc": best_val_acc_on_best_loss,
                "test_acc": best_test_acc_on_best_loss,
            },
        },
        ckpt_best_loss_path,
    )

    torch.save(
        {
            "state_dict": best_acc_state,
            "config": cfg,
            "dataset": ds,
            "metrics": {
                "val_acc": best_val_acc,
                "test_acc": best_test_acc_on_best_val_acc,
            },
        },
        ckpt_best_acc_path,
    )
    log.info(
        "best checkpoints: loss=%s acc=%s",
        ckpt_best_loss_path,
        ckpt_best_acc_path,
    )

    log.info(
        "dataset=%s%s precompute_wall_time_sec=%.3f training_cost_time_sec=%.3f",
        ds,
        split_suffix,
        precompute_wall_time_sec,
        training_cost_time_sec,
    )

    return {
        "val_acc": best_val_acc,
        "test_acc": best_test_acc_on_best_val_acc,
        "val_acc_at_best_val_loss": best_val_acc_on_best_loss,
        "test_acc_at_best_val_loss": best_test_acc_on_best_loss,
        "ckpt_path": ckpt_best_loss_path,
        "ckpt_best_loss_path": ckpt_best_loss_path,
        "ckpt_best_acc_path": ckpt_best_acc_path,
        "curves_path": curves_path,
        "precompute_wall_time_sec": precompute_wall_time_sec,
        "training_cost_time_sec": training_cost_time_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--processed-name", type=str, default="cora")
    parser.add_argument("--split-idx", type=int, default=0)
    args = parser.parse_args()

    setup_logging(file_name="train.log", subdir=args.dataset)
    log = get_logger("train")

    ds = _normalize_dataset_name(args.dataset)
    log.info(
        "start dataset=%s processed=%s split_idx=%s",
        ds,
        args.processed_name,
        args.split_idx,
    )

    cfg = default_lsgnn_config(ds)
    if ds in CONFIG_DATASETS:
        cfg.update(load_lsgnn_config_json(ds))
    cfg["split_idx"] = args.split_idx
    sync_ds_old_env_with_config(ds)

    t0 = time()
    metrics = train_single_run(ds, cfg, args.processed_name, log)
    elapsed = time() - t0
    log.info(
        "dataset=%s full_wall_time_sec=%.3f precompute_wall_time_sec=%.3f training_cost_time_sec=%.3f | best_val_loss ckpt -> val_acc=%.4f test_acc=%.4f | %s",
        ds,
        elapsed,
        metrics["precompute_wall_time_sec"],
        metrics["training_cost_time_sec"],
        metrics["val_acc_at_best_val_loss"],
        metrics["test_acc_at_best_val_loss"],
        metrics["ckpt_best_loss_path"],
    )
    log.info(
        "dataset=%s | best_val_acc ckpt -> val_acc=%.4f test_acc=%.4f | %s",
        ds,
        metrics["val_acc"],
        metrics["test_acc"],
        metrics["ckpt_best_acc_path"],
    )


if __name__ == "__main__":
    main()
