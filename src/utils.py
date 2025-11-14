import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import torch
import json
from typing import Dict, List, Tuple, Optional

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    median_ae = np.median(np.abs(y_true - y_pred))
    p95_error = np.percentile(np.abs(y_true - y_pred), 95)
    within_20_pct = np.mean(np.abs((y_true - y_pred) / y_true) <= 0.2) * 100

    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'median_ae': median_ae,
        'p95_error': p95_error,
        'within_20_pct': within_20_pct
    }


def compute_cost_structure_similarity(y_true_sub, y_pred_sub):
    from sklearn.metrics.pairwise import cosine_similarity

    true_ratios = y_true_sub / y_true_sub.sum(axis=1, keepdims=True)
    pred_ratios = y_pred_sub / y_pred_sub.sum(axis=1, keepdims=True)

    similarities = []
    for i in range(len(true_ratios)):
        sim = cosine_similarity(true_ratios[i:i+1], pred_ratios[i:i+1])[0, 0]
        similarities.append(sim)

    return np.array(similarities), np.mean(similarities)

def plot_prediction_scatter(y_true, y_pred, title='Prediction vs True',
                            save_path=None, show_metrics=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=30)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Prediction')

    if show_metrics:
        metrics = compute_metrics(y_true, y_pred)
        text = f"R² = {metrics['r2']:.4f}\n"
        text += f"MAE = {metrics['mae']:.2f}\n"
        text += f"MAPE = {metrics['mape']:.2f}%"
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_residual_distribution(y_true, y_pred, title='Residual Distribution',
                               save_path=None):
    residuals = y_pred.flatten() - y_true.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(residuals), color='blue', linestyle='--',
                    linewidth=2, label=f'Mean = {np.mean(residuals):.2f}')
    axes[0].set_xlabel('Residuals', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Residual Histogram', fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=13)
    axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def plot_feature_importance(feature_names, importance_values, top_n=20,
                            title='Feature Importance', save_path=None):

    indices = np.argsort(importance_values)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_values = importance_values[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_values)))

    ax.barh(range(len(top_features)), top_values,
            color=colors, edgecolor='black')

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def stratified_split_by_cost(X, y, test_size=0.15, random_state=42):
    from sklearn.model_selection import train_test_split

    total_cost = y.sum(axis=1)
    cost_bins = pd.qcut(total_cost, q=4, labels=False, duplicates='drop')

    return train_test_split(
        X, y, test_size=test_size,
        stratify=cost_bins,
        random_state=random_state
    )


def create_comorbidity_groups(comorbidity_features, names=None):
    if names is None:
        names = [f'C{i}' for i in range(comorbidity_features.shape[1])]

    labels = []
    for i in range(len(comorbidity_features)):
        active = [names[j] for j in range(len(names))
                  if comorbidity_features[i, j] > 0]
        if not active:
            label = 'None'
        elif len(active) == 1:
            label = active[0]
        else:
            label = '+'.join(active[:2])
        labels.append(label)

    return labels


def detect_outliers(data, method='iqr', threshold=3):
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold

    else:
        raise ValueError("method must be 'iqr' or 'zscore'")

def save_results_to_excel(results_dict, save_path, sheet_name='Results'):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results_dict, dict):
        df = pd.DataFrame([results_dict])
    else:
        df = results_dict

    df.to_excel(save_path, sheet_name=sheet_name, index=False)
    print(f"Saved results to: {save_path}")


def save_model_config(config_dict, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    print(f"Saved config to: {save_path}")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {seed}")


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.best_score = np.inf
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.best_score = -np.inf
            self.is_better = lambda new, best: new > best + min_delta

    def __call__(self, score):
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def bootstrap_ci(y_true, y_pred, metric_func,
                 n_bootstrap=1000, confidence=0.95):

    n_samples = len(y_true)
    metrics = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        metrics.append(metric_func(y_true_sample, y_pred_sample))

    metrics = np.array(metrics)
    alpha = 1 - confidence
    ci_lower = np.percentile(metrics, alpha/2 * 100)
    ci_upper = np.percentile(metrics, (1 - alpha/2) * 100)
    mean_metric = np.mean(metrics)

    return ci_lower, ci_upper, mean_metric


def compare_models_delong(y_true, pred1, pred2):
    from scipy.stats import pearsonr, norm

    corr1, _ = pearsonr(y_true.flatten(), pred1.flatten())
    corr2, _ = pearsonr(y_true.flatten(), pred2.flatten())

    n = len(y_true)
    z1 = 0.5 * np.log((1 + corr1) / (1 - corr1))
    z2 = 0.5 * np.log((1 + corr2) / (1 - corr2))

    se = np.sqrt(2 / (n - 3))
    z_score = (z1 - z2) / se
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

    return {
        'corr1': corr1,
        'corr2': corr2,
        'z_score': z_score,
        'p_value': p_value
    }

class Logger:
    def __init__(self, log_file='train.log'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - {pd.Timestamp.now()}\n")
            f.write("="*50 + "\n\n")

    def log(self, message):
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.randn(100, 1) * 1000 + 5000
    y_pred = y_true + np.random.randn(100, 1) * 500

    metrics = compute_metrics(y_true, y_pred)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_prediction_scatter(
        y_true, y_pred,
        title='Prediction vs True (Test)',
        save_path='test_scatter.png'
    )

    print("\nSaved test scatter plot as test_scatter.png")
