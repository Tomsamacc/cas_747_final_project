import torch
import torch.nn.functional as F
from torch import nn
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_mean


class LSGNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_nodes,
        hidden_channels=16,
        K=5,
        beta=1.0,
        gamma=0.5,
        dropout=0.5,
        method="norm2",
        num_reduce_layers=1,
        use_A_embedding=False,
        out_norm=True,
        out_mlp=False,
        use_irdc=True,
    ):
        super().__init__()

        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.dropout = dropout
        self.use_irdc = use_irdc
        self.out_norm = out_norm
        self.use_A_embedding = use_A_embedding

        self.dist_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

        self.alpha_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3 * K),
        )

        if num_reduce_layers == 1:
            reduce_dims = [(2 * K + 1, in_channels, hidden_channels)]
        elif num_reduce_layers > 1:
            reduce_dims = [(2 * K + 1, in_channels, 2 * hidden_channels)]
            for _ in range(num_reduce_layers - 2):
                reduce_dims.append((2 * K + 1, 2 * hidden_channels, 2 * hidden_channels))
            reduce_dims.append((2 * K + 1, 2 * hidden_channels, hidden_channels))
        else:
            raise ValueError("num_reduce_layers must be >= 1")

        self.reduce_layers = nn.ParameterList(
            [nn.Parameter(torch.zeros(shape)) for shape in reduce_dims]
        )
        self.reset_parameters()

        if use_A_embedding:
            self.A_mlp = nn.Sequential(
                nn.Linear(num_nodes, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            )
            final_nz = K + 2
        else:
            self.A_mlp = None
            final_nz = K + 1

        if out_mlp:
            self.out_head = nn.Sequential(
                nn.Linear(final_nz * hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, out_channels),
            )
        else:
            self.out_head = nn.Linear(final_nz * hidden_channels, out_channels)

    @torch.no_grad()
    def reset_parameters(self):
        for param in self.reduce_layers:
            nn.init.xavier_uniform_(param.data)

    def _pairwise_distance(self, x, edge_index):
        def _d(feat, src, tgt):
            if self.method == "cos":
                return (feat[src] * feat[tgt]).sum(dim=-1)
            elif self.method == "norm2":
                return torch.norm(feat[src] - feat[tgt], p=2, dim=-1)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        split_size = 10_000
        d_list = []
        for ei in edge_index.split(split_size, dim=-1):
            src_i, tgt_i = ei
            d_list.append(_d(x, src_i, tgt_i))
        d = torch.cat(d_list, dim=0).view(-1, 1)
        return torch.cat([d, d.square()], dim=-1)

    @torch.no_grad()
    def precompute_dist_and_prop(self, x, edge_index, filters):
        dist = self._pairwise_distance(x, edge_index)
        x_out = self._propagate(x, filters)
        return dist, x_out

    def _propagate(self, x, filters):
        filter_l, filter_h = filters
        x = x.to(torch.float32)

        x_L = matmul(filter_l, x)
        x_H = matmul(filter_h, x)
        out_L = [x_L]
        out_H = [x_H]

        x_L_sum = torch.zeros_like(x_L)
        x_H_sum = torch.zeros_like(x_H)

        for _ in range(1, self.K):
            if self.use_irdc:
                x_L_sum = x_L_sum + out_L[-1]
                x_H_sum = x_H_sum + out_H[-1]
                x_L = matmul(filter_l, (1.0 - self.gamma) * x - self.gamma * x_L_sum)
                x_H = matmul(filter_h, (1.0 - self.gamma) * x - self.gamma * x_H_sum)
            else:
                x_L = matmul(filter_l, x_L)
                x_H = matmul(filter_h, x_H)
            out_L.append(x_L)
            out_H.append(x_H)

        return torch.stack([x] + out_L + out_H, dim=0)

    def _local_sim(self, x, edge_index, dist):
        _, tgt = edge_index
        dist_val = self.dist_mlp(dist).view(-1)
        return scatter_mean(
            dist_val.cpu(), tgt.cpu(), dim=0, out=torch.zeros(x.size(0))
        ).to(x.device)

    def forward(self, x, edge_index, dist, x_out_l_h):
        device = x.device
        n = x.size(0)

        local_sim = self._local_sim(x, edge_index, dist)
        ls_and_sq = torch.stack([local_sim, local_sim.square()], dim=-1)

        alpha = self.alpha_mlp(ls_and_sq)
        alpha = alpha.view(n, self.K, 3)
        alpha_I = alpha[:, :, 0].t().unsqueeze(-1)
        alpha_L = alpha[:, :, 1].t().unsqueeze(-1)
        alpha_H = alpha[:, :, 2].t().unsqueeze(-1)

        out = x_out_l_h
        for reduce_layer in self.reduce_layers:
            out = torch.bmm(out, reduce_layer)
            out = F.normalize(out, p=2, dim=-1)
            out = F.relu(out)

        x0 = out[0, :, :]
        out_I = x0.expand(self.K, -1, -1)
        out_L = out[1 : self.K + 1, :, :]
        out_H = out[self.K + 1 :, :, :]

        fused = alpha_I * out_I + alpha_L * out_L + alpha_H * out_H

        if self.A_mlp is not None:
            A = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=torch.ones(edge_index.size(1), device=device),
                sparse_sizes=(n, n),
            ).to_torch_sparse_coo_tensor()
            A_embed = self.A_mlp(A)
            fused = torch.cat([x0.unsqueeze(0), fused, A_embed.unsqueeze(0)], dim=0)
        else:
            fused = torch.cat([x0.unsqueeze(0), fused], dim=0)

        if self.out_norm:
            fused = F.normalize(fused, p=2, dim=-1)
        fused = F.dropout(fused, p=self.dropout, training=self.training)

        fused = fused.permute(1, 0, 2).reshape(n, -1)
        return self.out_head(fused)
