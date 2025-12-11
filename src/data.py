# cap_preprocessor.py
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


class CAPDataPreprocessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.scalers = {}
        self.encoders = {}
        self.imputer = IterativeImputer(random_state=42)
        self.feature_names = None
        self.comorbidity_cols = None
        
        # === 特征分组（与模型完全对应）===
        self.demographic_cols = ['age', 'gender', 'BMI']
        self.laboratory_cols = [
            'Initial_WBC', 'Initial_Neutrophils', 'Initial_Hemoglobin', 'Initial_Platelets',
            'Initial_CRP', 'Initial_PCT', 'Initial_ESR', 'Initial_ALT', 'Initial_AST',
            'Initial_Albumin', 'Initial_Creatinine', 'Initial_BUN', 'Initial_Sodium',
            'Initial_Potassium', 'Initial_Chloride', 'T0_PaO2', 'T0_PaCO2', 'Initial_pH',
            'Initial_Lactate', 'Initial_PT', 'Initial_APTT', 'Initial_D_Dimer'
        ]
        self.pathogen_cols = [
            'Sputum_Culture', 'Blood_Culture', 'Viral_Test',
            'Fungal_Culture', 'Resistance_Analysis', 'Resistant_Infection'
        ]
        self.comorbidity_cols = [
            'Hypertension', 'Diabetes', 'Heart_Failure', 'CKD_Stage',
            'Chronic_Liver_Disease', 'Malignancy'
        ]
        self.treatment_cols = ['Expectorant', 'Bronchodilator', 'CBC_Count', 'LOS', 'ICU_Admission']
        self.target_cols = [
            'Total_Cost', 'Bed_Fee', 'Examination_Fee', 'Treatment_Fee',
            'Surgery_Fee', 'Nursing_Fee', 'Material_Fee', 'Other_Fee'
        ]

        # 病原学 & 耐药词汇表
        self.pathogen_vocab = {
            'Negative': 0, 'Streptococcus_pneumoniae': 1, 'Staphylococcus_aureus': 2,
            'Pseudomonas_aeruginosa': 3, 'Acinetobacter_baumannii': 4, 'Klebsiella_pneumoniae': 5,
            'Haemophilus_influenzae': 6, 'Candida_albicans': 7, 'Influenza_A': 8,
            'Influenza_B': 9, 'Other': 10, np.nan: 0
        }
        self.resistance_vocab = {
            'None': 0, 'MRSA': 1, 'ESBLs': 2, 'CRAB': 3, 'CRPA': 4, 'Other': 5, np.nan: 0
        }

    def _impute_and_encode(self):
        print("Step 1: MICE 多重插补 + 类别编码...")
        cat_cols = [col for col in self.pathogen_cols + self.comorbidity_cols if col in self.data.columns]
        num_cols = [col for col in self.demographic_cols + self.laboratory_cols + self.treatment_cols 
                   if col in self.data.columns and col != 'gender']

        # 临时编码类别变量
        for col in cat_cols:
            self.data[col] = self.data[col].astype(str)
            le = LabelEncoder()
            self.data[f"{col}_tmp"] = le.fit_transform(self.data[col])
            self.encoders[col] = le

        tmp_cols = [f"{col}_tmp" for col in cat_cols] + num_cols
        imputed = self.imputer.fit_transform(self.data[tmp_cols])
        
        for i, col in enumerate(tmp_cols):
            if col.endswith("_tmp"):
                orig_col = col.replace("_tmp", "")
                values = imputed[:, i]
                values = np.round(values).astype(int)
                le = self.encoders[orig_col]
                self.data[orig_col] = le.inverse_transform(values.clip(0, len(le.classes_)-1))
            else:
                self.data[col] = imputed[:, i]
        print(f"缺失值填补完成，剩余缺失: {self.data.isnull().sum().sum()}")

    def _encode_pathogens(self):
        print("Step 2: 病原学编码...")
        for col in ['Sputum_Culture', 'Blood_Culture', 'Viral_Test', 'Fungal_Culture']:
            if col in self.data.columns:
                self.data[f"{col}_id"] = self.data[col].map(self.pathogen_vocab).fillna(0).astype(int)
        if 'Resistance_Analysis' in self.data.columns:
            self.data["Resistance_Analysis_id"] = self.data['Resistance_Analysis'].map(self.resistance_vocab).fillna(0).astype(int)
        if 'Resistant_Infection' in self.data.columns:
            self.data["Resistant_Infection"] = self.data["Resistant_Infection"].fillna(0)

    def _create_clinical_features(self):
        print("Step 3: 创建临床衍生特征...")
        if 'T0_PaO2' in self.data.columns:
            self.data['PF_ratio'] = self.data['T0_PaO2'] / 0.21
        if all(c in self.data.columns for c in ['Initial_Neutrophils', 'Initial_WBC']) and self.data['Initial_WBC'].gt(0).any():
            self.data['Neutrophil_Percent'] = self.data['Initial_Neutrophils'] / self.data['Initial_WBC'] * 100
        if all(c in self.data.columns for c in ['Initial_Sodium', 'Initial_Potassium', 'Initial_Chloride']):
            self.data['Anion_Gap'] = self.data['Initial_Sodium'] - self.data['Initial_Chloride'] - self.data['Initial_Potassium']
        self.laboratory_cols += ['PF_ratio', 'Neutrophil_Percent', 'Anion_Gap']

    def _scale_numerical(self):
        print("Step 4: 标准化连续变量...")
        scale_cols = [c for c in self.demographic_cols + self.laboratory_cols + ['CBC_Count', 'LOS']
                     if c in self.data.columns and c != 'gender']
        for col in scale_cols:
            scaler = StandardScaler()
            self.data[col] = scaler.fit_transform(self.data[[col]]).ravel()
            self.scalers[col] = scaler
        print(f"已标准化 {len(self.scalers)} 个连续特征")

    def prepare_features(self):
        print("Step 5: 组装模型输入 (91维 + 6维共病)...")
        # 人口学
        demo_cols = ['age', 'gender', 'BMI']
        if 'gender' in self.data.columns:
            self.data['gender'] = (self.data['gender'] == 'Male').astype(float)  # 转为 0/1
        
        # 拼接所有特征
        feature_cols = (
            demo_cols +
            self.laboratory_cols +
            [c for c in self.data.columns if c.endswith('_id')] +
            self.comorbidity_cols +
            ['Expectorant', 'Bronchodilator', 'CBC_Count', 'LOS', 'ICU_Admission']
        )
        
        # 二值化共病（确保是0/1）
        for col in self.comorbidity_cols:
            if col in self.data.columns:
                self.data[col] = (self.data[col] != 'No').astype(float).fillna(0)
        
        X = self.data[feature_cols].fillna(0).values.astype(np.float32)
        y = self.data[self.target_cols].fillna(0).values.astype(np.float32)
        X_comorb = self.data[self.comorbidity_cols].values.astype(np.float32)
        
        self.feature_names = feature_cols
        print(f"最终输入维度: {X.shape[1]} (目标 91 维)")
        print(f"共病输入维度: {X_comorb.shape[1]} (6维)")
        
        return X, X_comorb, y

    def generate_table1(self, df=None, group_col=None, save_path=None):
        print("Generating Table 1 (发表级，带 p 值)...")
        if df is None: df = self.data.copy()
        rows = []
        
        cols_to_show = (
            ['age', 'gender', 'BMI'] +
            self.laboratory_cols[:10] + ['...'] +
            self.comorbidity_cols +
            ['LOS', 'ICU_Admission'] +
            self.pathogen_cols[:3]
        )
        cols_to_show = [c for c in cols_to_show if c in df.columns or c == '...']
        
        def fmt_num(x): 
            return f"{x.median():.1f} ({x.quantile(0.25):.1f}–{x.quantile(0.75):.1f})"
        def fmt_cat(x): 
            top = x.value_counts().index[0]
            pct = x.value_counts().max() / len(x) * 100
            return f"{top} ({pct:.1f}%)"
        
        if group_col is None:
            for col in cols_to_show:
                if col == '...': 
                    rows.append(["... (more lab indicators)", "..."])
                    continue
                s = df[col]
                rows.append([col, fmt_num(s) if s.dtype != 'object' else fmt_cat(s)])
            table = pd.DataFrame(rows, columns=["Variable", "Overall (N={})".format(len(df))])
        else:
            groups = sorted(df[group_col].unique())
            for col in cols_to_show:
                if col == '...': 
                    rows.append(["..."] + ["..."] * len(groups) + ["-"])
                    continue
                row = [col]
                values = [df[df[group_col]==g][col] for g in groups]
                for v in values:
                    row.append(fmt_num(v) if pd.api.types.is_numeric_dtype(v) else fmt_cat(v))
                # p value
                if pd.api.types.is_numeric_dtype(values[0]):
                    p = stats.mannwhitneyu(values[0], values[1], alternative='two-sided').pvalue
                else:
                    p = stats.chi2_contingency(pd.crosstab(df[group_col], df[col]))[1]
                row.append(f"{p:.3f}" if p >= 0.001 else "<0.001")
                rows.append(row)
            table = pd.DataFrame(rows, columns=["Variable"] + [f"{group_col}={g}" for g in groups] + ["p-value"])
        
        if save_path:
            table.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"Table 1 已保存 → {save_path}")
        return table

    def process(self):
        print("="*60)
        print(f"CAP 费用预测数据预处理启动 v{self.VERSION}")
        print("="*60)
        self._impute_and_encode()
        self._encode_pathogens()
        self._create_clinical_features()
        self._scale_numerical()
        X, X_comorb, y = self.prepare_features()
        print("数据预处理完成！")
        print("="*60)
        return {
            'X': X,
            'X_comorbidity': X_comorb,
            'y': y,
            'data': self.data.copy()
        }

    def split_stratified(self, data_dict, test_size=0.2, random_state=42):
        total_cost = data_dict['y'].sum(axis=1)
        quartiles = pd.qcut(total_cost, 4, labels=False, duplicates='drop')
        X_train, X_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
            data_dict['X'], data_dict['X_comorbidity'], data_dict['y'],
            test_size=test_size, stratify=quartiles, random_state=random_state
        )
        print(f"分层抽样完成 | 训练集: {len(X_train)} | 测试集: {len(X_test)}")
        return X_train, X_test, Xc_train, Xc_test, y_train, y_test

    def save(self, path="cap_preprocessor.pkl"):
        save_dict = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'comorbidity_cols': self.comorbidity_cols,
            'version': self.VERSION
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"预处理器已保存 → {path}")

    @staticmethod
    def load(path="cap_preprocessor.pkl"):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"预处理器加载成功 v{obj.get('version', '1.0')}")
        return obj


# 一键运行脚本
if __name__ == "__main__":
    preprocessor = CAPDataPreprocessor("data.csv")
    data = preprocessor.process()
    
    # 生成 Table 1
    preprocessor.generate_table1(save_path="Table1_overall.csv")
    
    # 分割数据
    (X_train, X_test, Xc_train, Xc_test, 
     y_train, y_test) = preprocessor.split_stratified(data)
    
    # 检查训练/测试集平衡性
    train_df = pd.DataFrame(X_train, columns=preprocessor.feature_names)
    train_df['split'] = 'train'
    test_df = pd.DataFrame(X_test, columns=preprocessor.feature_names)
    test_df['split'] = 'test'
    check_df = pd.concat([train_df, test_df])
    preprocessor.generate_table1(check_df, group_col='split', save_path="Table1_train_vs_test.csv")
    
    # 保存预处理器（上线部署必备）
    preprocessor.save("cap_preprocessor.pkl")
    
    print("\n所有任务完成！现在可以直接训练 ImprovedMultiTaskCostPredictor 了")
