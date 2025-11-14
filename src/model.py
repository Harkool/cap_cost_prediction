import torch
import torch.nn as nn

class ComorbidityGate(nn.Module):

    def __init__(self, comorbidity_dim=6, total_feature_dim=91):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(comorbidity_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, total_feature_dim),
            nn.Sigmoid()  
        )
    
    def forward(self, all_features, comorbidity_features):
        """
        Args:
            all_features: (batch, 91) 
            comorbidity_features: (batch, 6) 
        Returns:
            gated_features: (batch, 91) 
        """
        gate_weights = self.gate_network(comorbidity_features)
        return all_features * gate_weights


class MultiTaskCostPredictor(nn.Module):
    def __init__(
        self,
        input_dim=91,
        comorbidity_dim=6,
        shared_hidden_dims=[64, 32],
        task_head_dim=16,
        num_tasks=8,
        dropout_rate=0.4
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        self.comorbidity_gate = ComorbidityGate(
            comorbidity_dim=comorbidity_dim,
            total_feature_dim=input_dim
        )

        layers = []
        prev_dim = input_dim
        
        for hidden_dim in shared_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*layers)

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_hidden_dims[-1], task_head_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(task_head_dim, 1)
            )
            for _ in range(num_tasks)
        ])
        
        self.residual_correction = nn.Linear(num_tasks, 1)
        
    def forward(self, x, comorbidity_features):
        x_gated = self.comorbidity_gate(x, comorbidity_features)
        shared_features = self.shared_encoder(x_gated)  
        sub_costs = []
        for head in self.task_heads:
            cost = head(shared_features)  
            sub_costs.append(cost)
        
        sub_costs = torch.cat(sub_costs, dim=1)

        total_cost_base = sub_costs.sum(dim=1, keepdim=True) 
        residual = self.residual_correction(sub_costs) 
        total_cost = total_cost_base + residual
        
        return sub_costs, total_cost


if __name__ == "__main__":
    model = MultiTaskCostPredictor(
        input_dim=91,
        comorbidity_dim=6,
        shared_hidden_dims=[64, 32],
        task_head_dim=16,
        num_tasks=8,
        dropout_rate=0.4
    )
    
    batch_size = 16
    x_dummy = torch.randn(batch_size, 91)
    comorbidity_dummy = torch.randn(batch_size, 6)
    sub_costs, total_cost = model(x_dummy, comorbidity_dummy)
    print(f"Shape of sub-costs: {sub_costs.shape}")
    print(f"Shape of total cost: {total_cost.shape}")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
