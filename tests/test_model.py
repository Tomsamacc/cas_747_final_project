import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.models.model import LSGNN
from src.utils.helpers import build_l_h_filters


def run_smoke():
    num_nodes = 10
    in_dim = 8
    out_dim = 3

    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    model = LSGNN(
        in_channels=in_dim,
        out_channels=out_dim,
        num_nodes=num_nodes,
        hidden_channels=16,
        K=3,
    )

    filters = build_l_h_filters(edge_index, num_nodes, beta=1.0, transposed=True)
    with torch.no_grad():
        dist, x_out = model.precompute_dist_and_prop(x, edge_index, filters)
        out = model(x, edge_index, dist, x_out)

    assert out.shape == (num_nodes, out_dim)


if __name__ == "__main__":
    run_smoke()
    print("ok: model build + forward smoke")
