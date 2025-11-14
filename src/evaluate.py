import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from utils import (
    compute_metrics,
    plot_prediction_scatter,
    plot_residual_distribution,
    save_results_to_excel
)


class ModelEvaluator:

    def __init__(self, model, device='cpu', output_dir='./outputs'):
        """
        Args:
            model: trained model
            device: computation device
            output_dir: directory to store evaluation results
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fig_dir = self.output_dir / 'figures'
        self.result_dir = self.output_dir / 'results'
        self.fig_dir.mkdir(exist_ok=True)
        self.result_dir.mkdir(exist_ok=True)

        self.sub_task_names = [
            'Total_Medical_Cost', 'Bed_Fee', 'Examination_Fee', 'Treatment_Fee',
            'Surgery_Fee', 'Nursing_Fee', 'Material_Fee', 'Other_Fee'
        ]

    def evaluate_dataset(self, X_test, comorbidity_test, y_test, dataset_name='Test'):
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}")

        X_tensor = torch.FloatTensor(X_test).to(self.device)
        comorbidity_tensor = torch.FloatTensor(comorbidity_test).to(self.device)

        with torch.no_grad():
            pred_sub, pred_total = self.model(X_tensor, comorbidity_tensor)
            pred_sub = pred_sub.cpu().numpy()
            pred_total = pred_total.cpu().numpy()

        y_total_true = y_test.sum(axis=1, keepdims=True)

        print("\n[Total Cost Performance]")
        total_metrics = compute_metrics(y_total_true, pred_total)

        print(f"  R²:  {total_metrics['r2']:.4f}")
        print(f"  MAE: ¥{total_metrics['mae']:.2f}")
        print(f"  RMSE: ¥{total_metrics['rmse']:.2f}")
        print(f"  MAPE: {total_metrics['mape']:.2f}%")
        print(f"  Median AE: ¥{total_metrics['median_ae']:.2f}")

        print("\n[Sub-Cost Performance]")
        print(f"{'Item':<15} {'R²':>8} {'MAE(¥)':>10} {'MAPE(%)':>10}")
        print("-" * 45)

        sub_metrics_list = []
        for i, name in enumerate(self.sub_task_names):
            metrics = compute_metrics(y_test[:, i:i+1], pred_sub[:, i:i+1])
            sub_metrics_list.append(metrics)
            print(f"{name:<15} {metrics['r2']:>8.4f} {metrics['mae']:>10.2f} {metrics['mape']:>10.2f}")

        avg_sub_r2 = np.mean([m['r2'] for m in sub_metrics_list])
        avg_sub_mape = np.mean([m['mape'] for m in sub_metrics_list])

        print("-" * 45)
        print(f"{'Average':<15} {avg_sub_r2:>8.4f} {'':<10} {avg_sub_mape:>10.2f}")

        from sklearn.metrics import roc_auc_score

        print("\n[High-Cost Patient Identification]")
        for percentile in [10, 20]:
            threshold = np.percentile(y_total_true, 100 - percentile)
            y_high = (y_total_true >= threshold).astype(int).flatten()
            pred_score = pred_total.flatten()

            if len(np.unique(y_high)) > 1:
                auc = roc_auc_score(y_high, pred_score)
                print(f"  Top {percentile}% AUC: {auc:.4f}")

        print("\n[Cost Structure Similarity]")
        true_ratios = y_test / y_test.sum(axis=1, keepdims=True)
        pred_ratios = pred_sub / pred_sub.sum(axis=1, keepdims=True)

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = [
            cosine_similarity(true_ratios[i:i+1], pred_ratios[i:i+1])[0, 0]
            for i in range(len(true_ratios))
        ]

        avg_similarity = np.mean(similarities)
        print(f"  Cosine Similarity: {avg_similarity:.4f}")

        return {
            'dataset_name': dataset_name,
            'total_metrics': total_metrics,
            'sub_metrics': sub_metrics_list,
            'predictions': {
                'pred_sub': pred_sub,
                'pred_total': pred_total,
                'true_sub': y_test,
                'true_total': y_total_true
            },
            'structure_similarity': avg_similarity
        }

    def plot_overall_performance(self, results, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        pred_total = results['predictions']['pred_total'].flatten()
        true_total = results['predictions']['true_total'].flatten()

        # Scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(true_total, pred_total, alpha=0.5, s=30)

        min_val = min(true_total.min(), pred_total.min())
        max_val = max(true_total.max(), pred_total.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax1.fill_between([min_val, max_val],
                         [min_val*0.8, max_val*0.8],
                         [min_val*1.2, max_val*1.2],
                         alpha=0.2, color='gray')

        ax1.set_xlabel('True Total Cost (¥)')
        ax1.set_ylabel('Predicted Total Cost (¥)')
        ax1.set_title(f'Total Cost Prediction (R²={results["total_metrics"]["r2"]:.4f})')
        ax1.grid(alpha=0.3)

        # Residual histogram
        ax2 = axes[0, 1]
        residuals = pred_total - true_total
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--')

        ax2.set_xlabel('Prediction Error (¥)')
        ax2.set_title('Residual Distribution')
        ax2.grid(alpha=0.3)

        # Relative error plot
        ax3 = axes[1, 0]
        rel_err = np.abs((pred_total - true_total) / true_total) * 100
        ax3.scatter(true_total, rel_err, alpha=0.5, s=30)
        ax3.axhline(20, color='red', linestyle='--')
        ax3.axhline(10, color='orange', linestyle='--')

        ax3.set_xlabel('True Total Cost (¥)')
        ax3.set_ylabel('Relative Error (%)')
        ax3.set_title('Relative Error Distribution')
        ax3.grid(alpha=0.3)

        # Subtask R² bar chart
        ax4 = axes[1, 1]
        r2_scores = [m['r2'] for m in results['sub_metrics']]
        x_pos = np.arange(len(self.sub_task_names))
        bars = ax4.bar(x_pos, r2_scores, color='skyblue', edgecolor='black', alpha=0.8)

        for bar, score in zip(bars, r2_scores):
            ax4.text(bar.get_x() + bar.get_width()/2., score,
                     f'{score:.3f}', ha='center', va='bottom')

        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(self.sub_task_names, rotation=45, ha='right')
        ax4.set_ylabel('R² Score')
        ax4.set_ylim(0, 1.0)
        ax4.set_title('Sub-Cost Prediction Performance')
        ax4.grid(alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {save_path}")

        plt.show()
        return fig

    def plot_comorbidity_heatmap(self, results, comorbidity_features, save_path=None):
        true_sub = results['predictions']['true_sub']

        comorbidity_names = ['HTN', 'DM', 'HF', 'CKD', 'CLD', 'Cancer']
        labels = []

        for row in comorbidity_features:
            active = [comorbidity_names[j] for j in range(6) if row[j] > 0]
            if not active:
                labels.append('None')
            elif len(active) == 1:
                labels.append(active[0])
            else:
                labels.append('+'.join(active[:2]))

        df = pd.DataFrame(true_sub, columns=self.sub_task_names)
        df['Comorbidity'] = labels

        grouped = df.groupby('Comorbidity').mean()
        counts = df['Comorbidity'].value_counts()
        grouped = grouped.loc[counts[counts >= 5].index]

        grouped['Total'] = grouped.sum(axis=1)
        grouped = grouped.sort_values('Total', ascending=False).drop('Total', axis=1)
        grouped = grouped / 1000

        fig, ax = plt.subplots(figsize=(12, max(6, len(grouped) * 0.5)))
        sns.heatmap(grouped, annot=True, fmt='.1f', cmap='YlOrRd',
                    cbar_kws={'label': 'Average Cost (k¥)'}, linewidths=.5, ax=ax)

        ax.set_xlabel('Sub-Cost Type')
        ax.set_ylabel('Comorbidity Group')
        ax.set_title('Comorbidity-Cost Heatmap')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap: {save_path}")

        plt.show()
        return fig, grouped

    def plot_error_analysis(self, results, save_path=None):
        pred_total = results['predictions']['pred_total'].flatten()
        true_total = results['predictions']['true_total'].flatten()

        abs_err = np.abs(pred_total - true_total)
        rel_err = abs_err / true_total * 100

        top_n = 10
        worst_idx = np.argsort(abs_err)[-top_n:][::-1]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        ax1 = axes[0]
        x = np.arange(top_n)
        width = 0.35

        ax1.bar(x - width/2, true_total[worst_idx], width, label='True', color='skyblue')
        ax1.bar(x + width/2, pred_total[worst_idx], width, label='Predicted', color='salmon')

        for i, idx in enumerate(worst_idx):
            ax1.text(i, max(true_total[idx], pred_total[idx]) + 2000,
                     f'Err\n¥{abs_err[idx]:.0f}', ha='center', fontsize=8)

        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Case {i+1}' for i in range(top_n)])
        ax1.set_ylabel('Cost (¥)')
        ax1.set_title('Top 10 Worst Prediction Cases')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')

        ax2 = axes[1]
        quartiles = pd.qcut(true_total, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        data_by_quartile = [rel_err[quartiles == q] for q in quartiles.unique()]

        bp = ax2.boxplot(data_by_quartile, labels=quartiles.unique(),
                         patch_artist=True, showmeans=True)

        ax2.set_xlabel('True Cost Quartile')
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Prediction Error by Cost Level')
        ax2.grid(alpha=0.3, axis='y')
        ax2.axhline(20, color='red', linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved error analysis: {save_path}")

        plt.show()
        return worst_idx

    def shap_analysis(self, X_test, comorbidity_test, feature_names,
                      background_samples=100, save_dir=None):

        print(f"\n{'='*60}")
        print("Running SHAP analysis")
        print(f"{'='*60}")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if len(X_test) > background_samples:
            idx = np.random.choice(len(X_test), background_samples, replace=False)
            X_background = X_test[idx]
            comorbidity_background = comorbidity_test[idx]
        else:
            X_background = X_test
            comorbidity_background = comorbidity_test

        def model_wrapper(X):
            X_tensor = torch.FloatTensor(X).to(self.device)
            comorbidity_start = 56
            comorb = X[:, comorbidity_start:comorbidity_start+6]
            comorb_tensor = torch.FloatTensor(comorb).to(self.device)

            with torch.no_grad():
                _, pred_total = self.model(X_tensor, comorb_tensor)
            return pred_total.cpu().numpy()

        print("Building SHAP explainer...")
        explainer = shap.KernelExplainer(model_wrapper, X_background)

        n_explain = min(50, len(X_test))
        X_explain = X_test[:n_explain]

        print(f"Computing SHAP values for {n_explain} samples...")
        shap_values = explainer.shap_values(X_explain, nsamples=100)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-20:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(20),
                 mean_abs_shap[top_indices],
                 color=plt.cm.viridis(np.linspace(0, 1, 20)),
                 edgecolor='black')
        plt.yticks(range(20), [feature_names[i] for i in top_indices])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / 'shap_feature_importance.png', dpi=300)
        plt.show()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values[:, top_indices],
            X_explain[:, top_indices],
            feature_names=[feature_names[i] for i in top_indices],
            show=False
        )
        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / 'shap_summary_plot.png', dpi=300)
        plt.show()

        median_idx = np.argsort(model_wrapper(X_explain))[len(X_explain)//2]

        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[median_idx],
            X_explain[median_idx],
            feature_names=feature_names,
            max_display=15,
            show=False
        )
        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / 'shap_waterfall.png', dpi=300)
        plt.show()

        return {
            'shap_values': shap_values,
            'feature_importance': dict(zip(feature_names, mean_abs_shap))
        }

    def generate_report(self, results, save_path=None):
        if save_path is None:
            save_path = self.result_dir / 'evaluation_report.xlsx'

        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:

            df_total = pd.DataFrame([results['total_metrics']])
            df_total.index = ['TotalCost']
            df_total.to_excel(writer, sheet_name='Overall')

            df_sub = pd.DataFrame(results['sub_metrics'])
            df_sub.insert(0, 'SubCost', self.sub_task_names)
            df_sub.to_excel(writer, sheet_name='SubCosts', index=False)

            pred_df = pd.DataFrame({
                'True_Total': results['predictions']['true_total'][:100].flatten(),
                'Pred_Total': results['predictions']['pred_total'][:100].flatten(),
            })

            for i, name in enumerate(self.sub_task_names):
                pred_df[f'True_{name}'] = results['predictions']['true_sub'][:100, i]
                pred_df[f'Pred_{name}'] = results['predictions']['pred_sub'][:100, i]

            pred_df['Absolute_Error'] = np.abs(
                pred_df['Pred_Total'] - pred_df['True_Total']
            )
            pred_df['Relative_Error(%)'] = pred_df['Absolute_Error'] / pred_df['True_Total'] * 100

            pred_df.to_excel(writer, sheet_name='SamplePredictions', index=False)

        print(f"\nSaved evaluation report: {save_path}")
        return save_path


def main(model_path, data_path, output_dir='./outputs'):
    print("="*70)
    print("MODEL EVALUATION SYSTEM")
    print("="*70)

    from data_preprocessor import CAPDataPreprocessor

    print("\nStep 1: Loading Data")
    preprocessor = CAPDataPreprocessor(data_path)
    data_dict = preprocessor.process()
    splits = preprocessor.split_data(data_dict)

    comorbidity_start_idx = 56

    X_test = splits['X_test']
    comorbidity_test = X_test[:, comorbidity_start_idx:comorbidity_start_idx+6]
    y_test = splits['y_test']

    print("\nStep 2: Loading Model")
    from model import MultiTaskCostPredictor

    model = MultiTaskCostPredictor(
        input_dim=X_test.shape[1],
        comorbidity_dim=6,
        shared_hidden_dims=[64, 32],
        task_head_dim=16,
        num_tasks=8,
        dropout_rate=0.4
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Loaded model: {model_path}")

    print("\nStep 3: Initializing Evaluator")
    evaluator = ModelEvaluator(model, device='cpu', output_dir=output_dir)

    print("\nStep 4: Evaluating Model")
    results = evaluator.evaluate_dataset(X_test, comorbidity_test, y_test, 'Test')

    print("\nStep 5: Visualizations")
    evaluator.plot_overall_performance(
        results,
        save_path=evaluator.fig_dir / 'overall_performance.png'
    )
    evaluator.plot_comorbidity_heatmap(
        results,
        comorbidity_test,
        save_path=evaluator.fig_dir / 'comorbidity_heatmap.png'
    )
    evaluator.plot_error_analysis(
        results,
        save_path=evaluator.fig_dir / 'error_analysis.png'
    )

    print("\nStep 6: SHAP Analysis (Optional)")
    feature_names = (
        preprocessor.demographic_cols +
        preprocessor.laboratory_cols +
        preprocessor.pathogen_cols +
        preprocessor.comorbidity_cols +
        preprocessor.treatment_cols
    )

    user_input = input("Run SHAP analysis? [y/N]: ")
    if user_input.lower() == 'y':
        evaluator.shap_analysis(
            X_test,
            comorbidity_test,
            feature_names,
            background_samples=50,
            save_dir=evaluator.fig_dir / 'shap'
        )

    print("\nStep 7: Generating Report")
    report_path = evaluator.generate_report(results)

    print("\nEvaluation completed.")
    print(f"Figures saved in: {evaluator.fig_dir}")
    print(f"Report saved at: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CAP Cost Prediction Model Evaluation')
    parser.add_argument('--model_path', type=str, default='./outputs/models/best_model.pth')
    parser.add_argument('--data_path', type=str, default='./data/data.csv')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()
    main(args.model_path, args.data_path, args.output_dir)
