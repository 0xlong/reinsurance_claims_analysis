import os
import pickle
import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DB_PATH = os.path.join(DATA_DIR, 'insurance.db')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

def load_data():
    """Load the final pricing mart dataset from DuckDB."""
    print(f"Connecting to database at {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("SELECT * FROM mart_pricing").fetchdf()
    conn.close()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def build_preprocessor():
    """Build the feature preprocessing pipeline."""
    # Define categorical and numerical features
    categorical_features = ['area_code', 'vehicle_brand', 'fuel_type', 'region_code']
    numerical_features = ['vehicle_age', 'driver_age', 'bonus_malus', 'population_density', 'region_population']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='drop'
    )
    return preprocessor, categorical_features, numerical_features

def train_frequency_model(df, preprocessor, feature_cols):
    """Train the XGBoost Poisson Frequency Model."""
    print("\n--- Training Frequency Model (Poisson) ---")
    
    # Target: claim frequency rate (claim_count / exposure)
    y = (df['claim_count'] / df['exposure']).values
    weights = df['exposure'].values
    X_raw = df[feature_cols]
    
    # Fit the preprocessing pipeline on the data
    X_transformed = preprocessor.fit_transform(X_raw)
    
    # Retrieve clean feature names out of the transformer
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Train-test split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_df, y, weights, test_size=0.2, random_state=42
    )
    
    # Set base_score to the weighted average claim frequency (total claims / total exposure)
    # This aligns the starting prediction with the global portfolio rate
    portfolio_frequency = df['claim_count'].sum() / df['exposure'].sum()
    
    # XGBoost Regressor with Poisson Objective
    freq_model = xgb.XGBRegressor(
        objective='count:poisson',
        base_score=portfolio_frequency,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        random_state=42
    )
    
    print("Fitting model...")
    freq_model.fit(X_train, y_train, sample_weight=w_train)
    
    # Evaluate with Poisson deviance
    train_preds = freq_model.predict(X_train)
    test_preds = freq_model.predict(X_test)
    
    # Custom Poisson Deviance metric
    def poisson_deviance(y_true, y_pred, weights):
        # Prevent division by zero / negative logs
        y_pred = np.clip(y_pred, 1e-9, None)
        term1 = y_true * np.log(y_true / y_pred)
        term1 = np.where(y_true > 0, term1, 0)
        deviance = 2 * np.sum(weights * (term1 - (y_true - y_pred)))
        return deviance / np.sum(weights)
        
    print(f"Train Poisson Deviance: {poisson_deviance(y_train, train_preds, w_train):.5f}")
    print(f"Test Poisson Deviance: {poisson_deviance(y_test, test_preds, w_test):.5f}")
    
    # Combine preprocessor and model into a single Pipeline for easy deployment
    freq_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', freq_model)
    ])
    
    return freq_pipeline, X_df, freq_model

def train_severity_model(df, preprocessor, feature_cols):
    """Train the XGBoost Gamma Severity Model on policies with claims."""
    print("\n--- Training Severity Model (Gamma) ---")
    
    # Subset to policies with at least one claim and positive claim cost (Gamma requirement)
    df_claimed = df[(df['claim_count'] > 0) & (df['avg_claim_amount'] > 0)].copy()
    print(f"Filtered for severity modeling: {len(df_claimed)} claimed policies with positive cost.")
    
    # Target: avg_claim_amount
    y = df_claimed['avg_claim_amount'].values
    X_raw = df_claimed[feature_cols]
    
    # Fit preprocessor
    X_transformed = preprocessor.fit_transform(X_raw)
    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Set base_score to the mean of training severities (average claim amount)
    # This prevents gradient explosion/unstable splits due to the log-link function
    mean_severity = y_train.mean()
    
    # XGBoost Regressor with Gamma Objective
    sev_model = xgb.XGBRegressor(
        objective='reg:gamma',
        base_score=mean_severity,
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    
    print("Fitting model...")
    sev_model.fit(X_train, y_train)
    
    # Evaluate with Gamma deviance
    train_preds = sev_model.predict(X_train)
    test_preds = sev_model.predict(X_test)
    
    def gamma_deviance(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-9, None)
        deviance = 2 * np.sum(-np.log(y_true / y_pred) + (y_true - y_pred) / y_pred)
        return deviance / len(y_true)
        
    print(f"Train Gamma Deviance: {gamma_deviance(y_train, train_preds):.5f}")
    print(f"Test Gamma Deviance: {gamma_deviance(y_test, test_preds):.5f}")
    
    # Build complete deployment pipeline
    sev_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', sev_model)
    ])
    
    return sev_pipeline, X_df, sev_model

def generate_shap_plots(model, X_df, model_name):
    """Generate and save SHAP summary plots."""
    print(f"Generating SHAP plots for {model_name}...")
    
    # Sample a subset of data for fast SHAP computation
    sample_size = min(2000, len(X_df))
    X_sample = X_df.sample(n=sample_size, random_state=42)
    
    # XGBoost model SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Plot and save
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary Plot - {model_name.capitalize()} Model", fontsize=14, pad=15)
    plt.tight_layout()
    
    plot_path = os.path.join(MODELS_DIR, f"shap_{model_name}_summary.png")
    os.makedirs(MODELS_DIR, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP plot to {plot_path}")

def save_pipelines(freq_pipeline, sev_pipeline):
    """Save models as serialized pipelines."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    freq_path = os.path.join(MODELS_DIR, 'frequency_pipeline.pkl')
    sev_path = os.path.join(MODELS_DIR, 'severity_pipeline.pkl')
    
    with open(freq_path, 'wb') as f:
        pickle.dump(freq_pipeline, f)
    with open(sev_path, 'wb') as f:
        pickle.dump(sev_pipeline, f)
        
    print(f"\nSaved Frequency pipeline to {freq_path}")
    print(f"Saved Severity pipeline to {sev_path}")

if __name__ == "__main__":
    print("Starting Phase 2: Risk Pricing Model Pipeline...")
    
    # Load dataset
    df = load_data()
    
    # Select features
    feature_cols = [
        'area_code', 'vehicle_power', 'vehicle_age', 'driver_age', 
        'bonus_malus', 'vehicle_brand', 'fuel_type', 'population_density', 
        'region_code', 'region_population'
    ]
    
    # Set up scaling & encoding transformers
    preprocessor, _, _ = build_preprocessor()
    
    # Train Frequency model
    freq_pipeline, X_df_freq, freq_model = train_frequency_model(df, preprocessor, feature_cols)
    generate_shap_plots(freq_model, X_df_freq, "frequency")
    
    # Train Severity model
    # Re-instantiate the preprocessor to avoid state contamination
    preprocessor_sev, _, _ = build_preprocessor()
    sev_pipeline, X_df_sev, sev_model = train_severity_model(df, preprocessor_sev, feature_cols)
    generate_shap_plots(sev_model, X_df_sev, "severity")
    
    # Save the deployment pipelines
    save_pipelines(freq_pipeline, sev_pipeline)
    
    print("\nPhase 2 Model Pipeline completed successfully.")
