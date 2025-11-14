import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

class CostPredictionTrainer:
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=5e-4,
        weight_decay=1e-3
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.sub_task_weights = torch.tensor([
            0.25,  # Total medical cost
            0.12,  # Bed fee
            0.18,  # Examination fee
            0.20,  # Treatment fee
            0.05,  # Surgery fee
            0.08,  # Nursing fee
            0.08,  # Material fee
            0.04   # Other fees
        ]).to(device)
        
    def custom_loss(self, pred_sub, pred_total, true_sub, true_total, comorbidity):
        """
        Custom multitask loss function
        """
        sub_loss = 0
        for i in range(8):
            sub_loss += self.sub_task_weights[i] * nn.functional.mse_loss(
                pred_sub[:, i], true_sub[:, i]
            )
        
        total_loss = nn.functional.mse_loss(pred_total.squeeze(), true_total)

        pred_sum = pred_sub.sum(dim=1)
        consistency_loss = nn.functional.mse_loss(pred_sum, pred_total.squeeze())

        comorbidity_penalty = 0
        has_dm_ckd = (comorbidity[:, 1] == 1) | (comorbidity[:, 3] > 0)
        if has_dm_ckd.any():
            comorbidity_penalty += nn.functional.mse_loss(
                pred_sub[has_dm_ckd, 2], true_sub[has_dm_ckd, 2]
            )

        loss = sub_loss + 0.15 * total_loss + 0.1 * consistency_loss + 0.05 * comorbidity_penalty
        
        return loss, {
            'sub_loss': sub_loss.item(),
            'total_loss': total_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'comorbidity_penalty': comorbidity_penalty if isinstance(comorbidity_penalty, float) else comorbidity_penalty.item()
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            X, comorbidity, y = batch
            X = X.to(self.device)
            comorbidity = comorbidity.to(self.device)
            y = y.to(self.device)
            pred_sub, pred_total = self.model(X, comorbidity)
            true_total = y.sum(dim=1)
            loss, loss_dict = self.custom_loss(
                pred_sub, pred_total, y, true_total, comorbidity
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds_sub = []
        all_preds_total = []
        all_trues = []
        
        with torch.no_grad():
            for batch in val_loader:
                X, comorbidity, y = batch
                X = X.to(self.device)
                comorbidity = comorbidity.to(self.device)
                y = y.to(self.device)
                
                pred_sub, pred_total = self.model(X, comorbidity)
                
                true_total = y.sum(dim=1)
                loss, _ = self.custom_loss(
                    pred_sub, pred_total, y, true_total, comorbidity
                )
                
                total_loss += loss.item()
                all_preds_sub.append(pred_sub.cpu().numpy())
                all_preds_total.append(pred_total.cpu().numpy())
                all_trues.append(y.cpu().numpy())

        all_preds_sub = np.vstack(all_preds_sub)
        all_preds_total = np.vstack(all_preds_total)
        all_trues = np.vstack(all_trues)
        all_trues_total = all_trues.sum(axis=1, keepdims=True)
        ss_res = np.sum((all_trues_total - all_preds_total) ** 2)
        ss_tot = np.sum((all_trues_total - np.mean(all_trues_total)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        mape = np.mean(np.abs((all_trues_total - all_preds_total) / all_trues_total)) * 100
        return total_loss / len(val_loader), r2, mape
    
    def fit(self, train_loader, val_loader, epochs=150, patience=25):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, r2, mape = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
                print(f"  Best Val Loss: {best_val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("\n✅ Training complete! Best model loaded.")


if __name__ == "__main__":
    from data import CAPDataPreprocessor
    from model import MultiTaskCostPredictor
    print("Step 1: Loading and preprocessing data")
    preprocessor = CAPDataPreprocessor('data.csv')
    data_dict = preprocessor.process()
    splits = preprocessor.split_data(data_dict)
    print("\nStep 2: Building DataLoaders")
    comorbidity_start_idx = 56
    
    train_dataset = TensorDataset(
        torch.FloatTensor(splits['X_train']),
        torch.FloatTensor(splits['X_train'][:, comorbidity_start_idx:comorbidity_start_idx+6]),
        torch.FloatTensor(splits['y_train'])
    )
    
    val_dataset = TensorDataset(
        torch.FloatFloatTensor(splits['X_val']),
        torch.FloatTensor(splits['X_val'][:, comorbidity_start_idx:comorbidity_start_idx+6]),
        torch.FloatTensor(splits['y_val'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("\nStep 3: Initializing model")
    model = MultiTaskCostPredictor(
        input_dim=splits['X_train'].shape[1],
        comorbidity_dim=6,
        shared_hidden_dims=[64, 32],
        task_head_dim=16,
        num_tasks=8,
        dropout_rate=0.4
    )
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nStep 4: Training starts")
    trainer = CostPredictionTrainer(
        model,
        learning_rate=5e-4,
        weight_decay=1e-3
    )
    
    trainer.fit(train_loader, val_loader, epochs=150, patience=25)
    
    print("\n🎉 All steps completed!")
