from math import ceil
import numpy as np
import torch

HETERO_10_SPLIT_NAMES = frozenset(
    {"chameleon", "squirrel", "actor", "cornell", "texas", "wisconsin"}
)


def even_quantile_labels(vals, nclasses, verbose=False):
    vals = np.asarray(vals, dtype=np.float64).reshape(-1)
    label = -np.ones(vals.shape[0], dtype=np.int64)
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = float(np.nanquantile(vals, (k + 1) / nclasses))
        inds = (vals >= lower) & (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    if verbose:
        print("even_quantile_labels:", nclasses, "classes")
    return label


def stack_ogbn_arxiv_official_masks(num_nodes, split_idx, num_splits=10):
    train = torch.zeros(num_nodes, dtype=torch.bool)
    val = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)
    train[split_idx["train"]] = True
    val[split_idx["valid"]] = True
    test[split_idx["test"]] = True
    train = train.unsqueeze(1).expand(-1, num_splits).clone()
    val = val.unsqueeze(1).expand(-1, num_splits).clone()
    test = test.unsqueeze(1).expand(-1, num_splits).clone()
    return train, val, test


# For PyG not supporting multi-split train_mask, we balance the split
def class_balanced_split(
    y,
    num_nodes,
    num_classes,
    train_r=0.48,
    val_r=0.32,
    test_r=0.20,
    seed=28,
):
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6

    np.random.seed(seed)
    idx = torch.arange(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    rest_mask = torch.ones(num_nodes, dtype=torch.bool)

    unlabeled_mask = y.eq(-1)
    num_labeled = int(num_nodes - unlabeled_mask.sum())
    nc = num_classes
    if unlabeled_mask.any():
        nc -= 1

    for k in range(nc):
        k_mask = y.eq(k)
        num_nodes_k = int(k_mask.sum())
        if num_nodes_k == 0:
            continue
        num_train_k = ceil(num_nodes_k * train_r)
        idx_k = idx[k_mask]
        idx_k = idx_k[torch.randperm(idx_k.size(0))]
        train_mask[idx_k[:num_train_k]] = True

    num_val = round(val_r * num_labeled)
    rest_mask[train_mask] = False
    rest_mask[unlabeled_mask] = False
    rest_idx = idx[rest_mask]
    rest_idx = rest_idx[torch.randperm(rest_idx.size(0))]

    val_idx = rest_idx[:num_val]
    test_idx = rest_idx[num_val:]
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def build_lsgnn_split_stack(
    data,
    dataset_name,
    num_splits=10,
    global_seed_base=28,
    train_r=0.48,
    val_r=0.32,
    test_r=0.20,
):
    name = dataset_name.lower()
    y = data.y.squeeze()
    n = y.size(0)
    num_classes = int(y.unique().shape[0])

    if name in HETERO_10_SPLIT_NAMES:
        if not hasattr(data, "train_mask") or data.train_mask is None:
            raise ValueError(f"{name} expected multi-split train_mask from PyG")
        tm = data.train_mask
        if tm.dim() == 1:
            raise ValueError(f"{name}: got 1D mask; expected 10-split Geom-GCN style data")
        s = min(num_splits, tm.size(1))
        # first s columns from PyG
        train_mask = tm[:, :s].bool()
        val_mask = data.val_mask[:, :s].bool()
        test_mask = data.test_mask[:, :s].bool()
        return train_mask, val_mask, test_mask

    tr_list, va_list, te_list = [], [], []
    for i in range(num_splits):
        tr, va, te = class_balanced_split(
            y, n, num_classes, train_r, val_r, test_r, seed=global_seed_base + i
        )
        tr_list.append(tr)
        va_list.append(va)
        te_list.append(te)
    # Planetoid-style: seeds 28, 29, ...
    return (
        torch.stack(tr_list, dim=1),
        torch.stack(va_list, dim=1),
        torch.stack(te_list, dim=1),
    )
