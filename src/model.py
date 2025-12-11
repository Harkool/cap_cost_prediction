import torch
import torch.nn as nn
import torch.nn.functional as F


class ComorbidityGate(nn.Module):
    def __init__(self, comorbidity_dim=6, feature_dim=91, hidden=64):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(comorbidity_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(comorbidity_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim)
        )
        for m in [self.gamma_net, self.beta_net]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, comorb):
        # x: (B, 91), comorb: (B, 6)
        gamma = self.gamma_net(comorb)
        beta  = self.beta_net(comorb) 
        return x * (1 + gamma) + beta

class MultiTaskCostPredictor(nn.Module):
    def __init__(
        self,
        input_dim=91,
        comorbidity_dim=6,
        num_tasks=8,
        shared_dims=[256, 128, 64, 32],
        task_head_dim=32,
        dropout=0.3
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.film_gate = ComorbidityGate(comorbidity_dim, input_dim)
        layers = []
        prev_dim = input_dim
        for i, h in enumerate(shared_dims):
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            if i < len(shared_dims) - 1 and h == shared_dims[i + 1]:
                layers.append(nn.Identity())
            prev_dim = h
        self.shared_encoder = nn.Sequential(*layers)
        self.residual_projs = nn.ModuleList()
        cur_dim = input_dim
        for h in shared_dims:
            if cur_dim != h:
                self.residual_projs.append(nn.Linear(cur_dim, h, bias=False))
            else:
                self.residual_projs.append(nn.Identity())
            cur_dim = h
        self.task_embedding = nn.Embedding(num_tasks, shared_dims[-1])
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dims[-1], task_head_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.6),
                nn.Linear(task_head_dim, 1)
            ) for _ in range(num_tasks)
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, comorbidity_features):
        x = self.film_gate(x, comorbidity_features)
        h = x
        for layer, proj in zip(self.shared_encoder, self.residual_projs):
            if isinstance(layer, nn.Identity):  
                continue
            h_new = layer(h)
            h = h_new + proj(h) if h.shape == h_new.shape else h_new

        shared_feat = h  # (B, last_dim)
        sub_costs = []
        for i, head in enumerate(self.task_heads):
            task_emb = self.task_embedding.weight[i]
            task_emb = task_emb.expand_as(shared_feat)
            feat_i = shared_feat + task_emb
            cost_raw = head(feat_i).squeeze(-1)
            cost = F.softplus(cost_raw)
            sub_costs.append(cost)

        sub_costs = torch.stack(sub_costs, dim=1)
        total_cost = sub_costs.sum(dim=1, keepdim=True)

        return sub_costs, total_cost

if __name__ == "__main__":
    model = MultiTaskCostPredictor(
        input_dim=91,
        comorbidity_dim=6,
        num_tasks=8,
        shared_dims=[256, 128, 64, 32],
        task_head_dim=32,
        dropout=0.3
    )

    B = 16
    x = torch.randn(B, 91)
    comorb = torch.randn(B, 6)

    sub_costs, total_cost = model(x, comorb)

    print(f"Sub-costs shape: {sub_costs.shape}")          # [16, 8]
    print(f"Total cost shape: {total_cost.shape}")        # [16, 1]
    print(f"All costs ≥ 0: {(sub_costs >= 0).all().item()}")
    print(f"Additivity check: {(sub_costs.sum(1) - total_cost.squeeze(1)).abs().max().item():.6f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
