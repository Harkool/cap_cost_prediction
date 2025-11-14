import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

class CAPDataPreprocessor:
    def __init__(self, csv_path):
        """
        Initialize the data preprocessor
        Args:
            csv_path: Path to the CSV file
        """
        self.data = pd.read_csv(csv_path)
        self.scalers = {}
        self.encoders = {}
        
        # Define feature columns
        self.demographic_cols = ['age', 'gender', 'BMI']
        
        self.laboratory_cols = [
            'Initial_WBC', 'Initial_Neutrophils', 'Initial_Hemoglobin', 'Initial_Platelets',
            'Initial_CRP', 'Initial_PCT', 'Initial_ESR', 'Initial_ALT', 'Initial_AST', 'Initial_Albumin',
            'Initial_Creatinine', 'Initial_BUN', 'Initial_Sodium', 'Initial_Potassium', 'Initial_Chloride',
            'T0_PaO2', 'T0_PaCO2', 'Initial_pH', 'Initial_Lactate',
            'Initial_PT', 'Initial_APTT', 'Initial_D_Dimer'
        ]
        
        self.pathogen_cols = [
            'Sputum_Culture', 'Blood_Culture', 'Viral_Test',
            'Fungal_Culture', 'Resistance_Analysis', 'Resistant_Infection'
        ]
        
        self.comorbidity_cols = [
            'Hypertension', 'Diabetes', 'Heart_Failure', 'CKD_Stage', 'Chronic_Liver_Disease', 'Malignancy'
        ]
        
        self.treatment_cols = [
            'Expectorant', 'Bronchodilator', 'CBC_Count', 'LOS', 'ICU_Admission'
        ]
        
        self.target_cols = [
            'Total_Cost', 'Bed_Fee', 'Examination_Fee', 'Treatment_Fee',
            'Surgery_Fee', 'Nursing_Fee', 'Material_Fee', 'Other_Fee'
        ]
        
    def handle_missing_values(self):
        """
        Handle missing values using Multiple Imputation (MICE)
        Numeric + Categorical supported
        """
        print("Handling missing values with Multiple Imputation (MICE)...")
        
        numeric_cols = [
            col for col in self.data.columns 
            if self.data[col].dtype != "object" and col in (
                self.demographic_cols + 
                self.laboratory_cols + 
                self.treatment_cols
            )
        ]
        
        categorical_cols = [
            col for col in self.data.columns
            if self.data[col].dtype == "object" and col in (
                self.pathogen_cols + self.comorbidity_cols
            )
        ]
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = self.data[col].astype(str)
            self.data[col] = le.fit_transform(self.data[col])
            self.encoders[col] = le
        imputer = IterativeImputer(random_state=42)
        imputed = imputer.fit_transform(self.data[numeric_cols + categorical_cols])
        imputed_df = pd.DataFrame(imputed, columns=numeric_cols + categorical_cols)
        for col in categorical_cols:
            le = self.encoders[col]
            imputed_df[col] = imputed_df[col].round().astype(int) 
            imputed_df[col] = le.inverse_transform(imputed_df[col])
        self.data[numeric_cols + categorical_cols] = imputed_df

        print("Multiple imputation completed.")
        print(f"Remaining missing values: {self.data.isnull().sum().sum()}")
    
    def encode_pathogen_features(self):
        """
        Encode pathogen-related features
        Create embedding indices for each culture result
        (If you have encoded before, you can skip it)
        """
        print("Encoding pathogen features...")
        
        # Pathogen vocabulary
        pathogen_vocab = {
            'Negative': 0,
            'Streptococcus_pneumoniae': 1,
            'Staphylococcus_aureus': 2,
            'Pseudomonas_aeruginosa': 3,
            'Acinetobacter_baumannii': 4,
            'Klebsiella_pneumoniae': 5,
            'Haemophilus_influenzae': 6,
            'Candida_albicans': 7,
            'Influenza_A': 8,
            'Influenza_B': 9,
            'Other': 10
        }
        
        resistance_vocab = {
            'None': 0,
            'MRSA': 1,
            'ESBLs': 2,
            'CRAB': 3,  # Carbapenem-resistant Acinetobacter baumannii
            'CRPA': 4,  # Carbapenem-resistant Pseudomonas aeruginosa
            'Other': 5
        }
        

        for col in ['Sputum_Culture', 'Blood_Culture', 'Viral_Test', 'Fungal_Culture']:
            if col in self.data.columns:
                self.data[f'{col}_encoded'] = self.data[col].map(
                    lambda x: pathogen_vocab.get(x, 10)
                )

        if 'Resistance_Analysis' in self.data.columns:
            self.data['Resistance_Analysis_encoded'] = self.data['Resistance_Analysis'].map(
                lambda x: resistance_vocab.get(x, 5)
            )
        
        self.pathogen_vocab_size = len(pathogen_vocab)
        self.resistance_vocab_size = len(resistance_vocab)
        
        print(f"Pathogen encoding completed: Pathogen vocab size = {self.pathogen_vocab_size}, Resistance vocab size = {self.resistance_vocab_size}")
    
    def create_derived_features(self):
        """Create derived features"""
        print("Creating derived features...")
        
        if 'T0_PaO2' in self.data.columns:
            self.data['PF_ratio'] = self.data['T0_PaO2'] / 0.21
        
        if all(col in self.data.columns for col in ['Initial_Neutrophils', 'Initial_WBC']):
            self.data['Neutrophil_Percent'] = (
                self.data['Initial_Neutrophils'] / self.data['Initial_WBC'] * 100
            )
        
        if all(col in self.data.columns for col in ['Initial_Sodium', 'Initial_Potassium', 'Initial_Chloride']):
            self.data['AG'] = self.data['Initial_Sodium'] - self.data['Initial_Chloride'] - (self.data['Initial_Potassium'] * 0.5)
        self.laboratory_cols.extend(['PF_ratio', 'Neutrophil_Percent', 'AG'])
        
        print(f"Derived features created: 3 new features added")
    
    def normalize_features(self):
        """Standardize continuous features"""
        print("Normalizing features...")
        
        normalize_cols = (
            self.demographic_cols +
            self.laboratory_cols +
            [col for col in self.treatment_cols if col in ['CBC_Count', 'LOS']]
        )
        
        for col in normalize_cols:
            if col in self.data.columns and col != 'gender':
                scaler = StandardScaler()
                self.data[f'{col}_scaled'] = scaler.fit_transform(
                    self.data[[col]]
                )
                self.scalers[col] = scaler
        
        print(f"Feature normalization completed. {len(self.scalers)} features processed.")
    
    def prepare_model_inputs(self):
        """
        Prepare model inputs
        Returns:
            X_dict: dictionary containing all modalities
            y: 8 cost components
        """
        print("Preparing model inputs...")
        
        demographic_features = []
        for col in self.demographic_cols:
            if f'{col}_scaled' in self.data.columns:
                demographic_features.append(f'{col}_scaled')
            else:
                demographic_features.append(col)
        
        X_demographic = self.data[demographic_features].values
        
        lab_features = []
        for col in self.laboratory_cols:
            if f'{col}_scaled' in self.data.columns:
                lab_features.append(f'{col}_scaled')
            else:
                lab_features.append(col)
        
        X_laboratory = self.data[lab_features].values
        
        pathogen_encoded_cols = [
            'Sputum_Culture_encoded', 'Blood_Culture_encoded', 'Viral_Test_encoded',
            'Fungal_Culture_encoded', 'Resistance_Analysis_encoded', 'Resistant_Infection'
        ]
        X_pathogen = self.data[pathogen_encoded_cols].values
        
        X_comorbidity = self.data[self.comorbidity_cols].values
        
        treatment_features = []
        for col in self.treatment_cols:
            if f'{col}_scaled' in self.data.columns:
                treatment_features.append(f'{col}_scaled')
            else:
                treatment_features.append(col)
        
        X_treatment = self.data[treatment_features].values
        
        y = self.data[self.target_cols].values
        
        X_all = np.concatenate([
            X_demographic,
            X_laboratory,
            X_pathogen,
            X_comorbidity,
            X_treatment
        ], axis=1)
        
        print(f"Input preparation completed:")
        print(f"  - Demographic dim: {X_demographic.shape[1]}")
        print(f"  - Laboratory dim: {X_laboratory.shape[1]}")
        print(f"  - Pathogen dim: {X_pathogen.shape[1]}")
        print(f"  - Comorbidity dim: {X_comorbidity.shape[1]}")
        print(f"  - Treatment dim: {X_treatment.shape[1]}")
        print(f"  - Total feature dim: {X_all.shape[1]}")
        print(f"  - Target dim: {y.shape[1]}")
        
        return {
            'X_all': X_all,
            'X_demographic': X_demographic,
            'X_laboratory': X_laboratory,
            'X_pathogen': X_pathogen,
            'X_comorbidity': X_comorbidity,
            'X_treatment': X_treatment,
            'y': y
        }

    def generate_table1(self, df=None, group_col=None, save_path=None):
        """
        Universal Table 1 generator
        - df: a DataFrame (e.g. train/val/test). If None → use full self.data
        - group_col: group variable. If None → no grouping (overall table)
        - save_path: export to CSV
        """

        if df is None:
            df = self.data.copy()

        print("Generating Table 1...")

        table_entries = []

        def summarize_numeric(series):
            median = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            return f"{median:.1f} ({iqr:.1f})"

        def summarize_categorical(series):
            vc = series.value_counts()
            top = vc.idxmax()
            perc = vc.max() / len(series) * 100
            return f"{top} ({perc:.1f}%)"

        features = (
            self.demographic_cols +
            self.laboratory_cols +
            self.comorbidity_cols +
            self.treatment_cols +
            self.pathogen_cols
        )
        features = [col for col in features if col in df.columns]


        if group_col is None:
            for col in features:
                s = df[col]
                if pd.api.types.is_numeric_dtype(s):
                    summary = summarize_numeric(s)
                else:
                    summary = summarize_categorical(s)
                table_entries.append([col, summary])

            table = pd.DataFrame(table_entries, columns=["Variable", "Overall"])

        else:
            groups = sorted(df[group_col].dropna().unique())
            for col in features:
                row = [col]
                for g in groups:
                    s = df[df[group_col] == g][col]
                    if pd.api.types.is_numeric_dtype(s):
                        summary = summarize_numeric(s)
                    else:
                        summary = summarize_categorical(s)
                    row.append(summary)
                table_entries.append(row)

            table = pd.DataFrame(
                table_entries,
                columns=["Variable"] + [f"{group_col}={g}" for g in groups]
            )

        if save_path:
            table.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"Table 1 saved to {save_path}")

        return table

    def process(self):
        """Complete preprocessing pipeline"""
        print("="*50)
        print("Starting data preprocessing")
        print("="*50)      
        self.handle_missing_values()
        self.encode_pathogen_features()
        self.create_derived_features()
        self.normalize_features()
        data_dict = self.prepare_model_inputs()        
        print("="*50)
        print("Data preprocessing completed!")
        print("="*50)        
        return data_dict
    
    def split_data(self, data_dict, test_size=0.2, random_state=42):
        """
        Split dataset into train and test sets (stratified by cost quartiles)
        """
        X_all = data_dict['X_all']
        y = data_dict['y']

        total_cost = y.sum(axis=1)
        cost_bins = pd.qcut(total_cost, q=4, labels=False)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y,
            test_size=test_size,
            stratify=cost_bins,
            random_state=random_state
        )
        print("\nDataset split:")
        print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_all)*100:.1f}%)")
        print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_all)*100:.1f}%)")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test
        }

if __name__ == "__main__":
    preprocessor = CAPDataPreprocessor('data.csv')
    data_dict = preprocessor.process()
    table1 = preprocessor.generate_table1("table1.csv")

    splits = preprocessor.split_data(data_dict)
    train_df = pd.DataFrame(splits["X_train"])
    test_df = pd.DataFrame(splits["X_test"])
    train_df["dataset"] = "train"
    test_df["dataset"] = "test"
    combined_df = pd.concat([train_df, test_df])
    table1_compare = preprocessor.generate_table1(combined_df,group_col="dataset",save_path="table1_train_vs_test.csv")
    print(table1_compare)
    print("\n✅ Data preparation completed. Ready for model training!")
