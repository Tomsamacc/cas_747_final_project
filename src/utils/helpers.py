import os
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
import matplotlib.pyplot as plt


def plot_training_curves(epochs,train_loss,val_loss,val_acc,test_acc,save_path,suptitle=None):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax0.plot(epochs, train_loss, label="train loss")
    ax0.plot(epochs, val_loss, label="val loss")
    ax0.set_ylabel("loss")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1.plot(epochs, val_acc, label="val acc")
    ax1.plot(epochs, test_acc, label="test acc")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.0)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)

    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path

def set_seed(seed=28):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scipy_coo_to_torch_sparse(sparse_mx):
    indices = torch.from_numpy(np.vstack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_l_h_filters(edge_index, num_nodes, beta=1.0, transposed=True):
    device = edge_index.device
    edge_index = edge_index.cpu()
    n = num_nodes

    edge_index, _ = remove_self_loops(edge_index)
    data = np.ones(edge_index.size(1), dtype=np.float32)
    if transposed:
        adj = sp.csr_matrix((data, (edge_index[0].numpy(), edge_index[1].numpy())), shape=(n, n))
    else:
        adj = sp.csr_matrix((data, (edge_index[1].numpy(), edge_index[0].numpy())), shape=(n, n))

    edge_index_sl, _ = add_remaining_self_loops(edge_index)
    data_sl = np.ones(edge_index_sl.size(1), dtype=np.float32)
    if transposed:
        adj_sl = sp.csr_matrix((data_sl, (edge_index_sl[0].numpy(), edge_index_sl[1].numpy())), shape=(n, n))
    else:
        adj_sl = sp.csr_matrix((data_sl, (edge_index_sl[1].numpy(), edge_index_sl[0].numpy())), shape=(n, n))

    deg = np.array(adj_sl.sum(axis=1)).flatten()
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float("inf")] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    identity = sp.eye(n)
    # D^{-1/2} A D^{-1/2} on edges + self-loops
    dad = deg_sqrt_inv * adj * deg_sqrt_inv
    filter_l = sp.coo_matrix(beta * identity + dad)
    filter_h = sp.coo_matrix((1.0 - beta) * identity - dad)

    filter_l = scipy_coo_to_torch_sparse(filter_l)
    filter_h = scipy_coo_to_torch_sparse(filter_h)

    filter_l = SparseTensor.from_torch_sparse_coo_tensor(filter_l).to(device)
    filter_h = SparseTensor.from_torch_sparse_coo_tensor(filter_h).to(device)
    return filter_l, filter_h


def accuracy(logits, y, mask):
    preds = logits.argmax(dim=1)
    correct = preds[mask].eq(y[mask]).float().sum()
    return (correct / mask.sum()).item()
