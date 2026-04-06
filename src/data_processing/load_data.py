from __future__ import annotations

import json
import os

import torch
from torch_geometric.datasets import Actor, Planetoid, WebKB, WikipediaNetwork
from torch_geometric.transforms import ToUndirected

from src.data_processing.paths import RAW_DIR
from src.data_processing.splits import even_quantile_labels


def _normalize_dataset_name(name: str) -> str:
    s = name.lower().replace("_", "-")
    if s == "arxiv_year":
        s = "arxiv-year"
    return s


def sync_ds_old_env_with_config(dataset_name: str) -> None:
    name = _normalize_dataset_name(dataset_name)
    cfg_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "configs", f"{name}.json")
    )
    ds_old = False
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            ds_old = bool(json.load(f).get("ds_old"))
    os.environ["DS_OLD"] = "1" if ds_old else "0"

# OGB PygNodePropPredDataset uses torch.load on processed data.pt; PyTorch 2.6+ defaults to
# weights_only=True and breaks on PyG objects. NodePropPredDataset avoids that path.


def _ogbn_arxiv_graph_to_data(root, undirected, relabel_year_quantiles=False):
    from ogb.nodeproppred import NodePropPredDataset
    from torch_geometric.data import Data

    ds = NodePropPredDataset(root=root, name="ogbn-arxiv")
    graph, label = ds[0]
    split_idx = ds.get_idx_split()
    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
    x = torch.as_tensor(graph["node_feat"], dtype=torch.float)
    y = torch.as_tensor(label, dtype=torch.long)
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    data = Data(x=x, edge_index=edge_index, y=y)
    if relabel_year_quantiles:
        ny = torch.as_tensor(graph["node_year"], dtype=torch.float).view(-1).numpy()
        yq = torch.as_tensor(even_quantile_labels(ny, 5, verbose=False), dtype=torch.long).unsqueeze(-1)
        data.y = yq
    if undirected:
        data = ToUndirected()(data)
    return data, split_idx


# PyG folder names (not .capitalize() for citeseer/pubmed)
PNAME = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
SUFFIX = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")


def _ok_raw(d, pyg_name):
    t = pyg_name.lower()
    return all(os.path.isfile(os.path.join(d, f"ind.{t}.{s}")) for s in SUFFIX)


def _fix_cased_dir(root, pyg_name):
    cap = os.path.join(root, pyg_name)
    low = os.path.join(root, pyg_name.lower())
    r_cap, r_low = os.path.join(cap, "raw"), os.path.join(low, "raw")
    if _ok_raw(r_cap, pyg_name):
        return
    if not _ok_raw(r_low, pyg_name):
        if os.environ.get("LSGNN_ALLOW_DOWNLOAD", "1").lower() in ("0", "false", "no"):
            raise FileNotFoundError(f"need planetoid files under {r_cap} or {r_low}")
        return
    if os.path.lexists(cap) and not os.path.islink(cap) and not _ok_raw(r_cap, pyg_name):
        raise RuntimeError(f"remove or fill {cap}; good data is under {r_low}")
    if not os.path.lexists(cap):
        os.symlink(os.path.abspath(low), cap, target_is_directory=True)


def load_graph(name, raw_root=None, undirected=False):
    name = name.lower().replace("_", "-")
    root = os.path.expanduser(raw_root) if raw_root else RAW_DIR

    if name in PNAME:
        pyg = PNAME[name]
        _fix_cased_dir(root, pyg)
        ds = Planetoid(root=root, name=pyg)
    elif name in ("chameleon", "squirrel"):
        ds = WikipediaNetwork(root=root, name=name)
    elif name in ("cornell", "texas", "wisconsin"):
        webkb_root = os.path.join(root, "_old") if int(os.environ.get("DS_OLD", "0")) else root
        ds = WebKB(root=webkb_root, name=name)
    elif name == "actor":
        ds = Actor(root=os.path.join(root, "Actor"))
    elif name == "ogbn-arxiv":
        data, split_idx = _ogbn_arxiv_graph_to_data(root, undirected, relabel_year_quantiles=False)
        data._ogb_split_idx = split_idx
        return data
    elif name == "arxiv-year":
        data, _split_idx = _ogbn_arxiv_graph_to_data(root, undirected, relabel_year_quantiles=True)
        return data
    else:
        raise ValueError(name)

    data = ds[0]
    if undirected:
        data = ToUndirected()(data)
    return data
