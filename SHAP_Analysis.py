import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import shap

# defines and creates directories
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'Final_Dataframe.csv')
results_dir = os.path.join(base_dir, "SHAP_Results")
plots_dir = os.path.join(results_dir, "SHAP_Plots")
os.makedirs(plots_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.dropna(inplace=True)

# Lag feature creation
def create_lag_features(df, selected_features, sequence_length=30):
    df_feat = df[['Date'] + selected_features].copy()
    lagged_data = []
    for lag in range(1, sequence_length + 1):
        shifted = df_feat[selected_features].shift(lag).add_suffix(f'_lag{lag}')
        lagged_data.append(shifted)
    df_lagged = pd.concat([df_feat] + lagged_data, axis=1)
    df_lagged.dropna(inplace=True)
    X = df_lagged.drop(columns=['Date'])
    y = df.loc[df_lagged.index, 'WTI']
    dates = df_lagged['Date']
    return X, y, dates

# selects features
selected_features = ['GPR', 'AMEX', 'INV', 'OVX', 'MSCI']
X_raw, y, dates = create_lag_features(df, selected_features)

# Train-test split before scaling
split_idx = int(0.8 * len(X_raw))
X_train_raw, X_test_raw = X_raw.iloc[:split_idx], X_raw.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = dates.iloc[split_idx:]

# Scale after split (to avoid leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# models defined
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# SHAP analysis
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Create output folder
    model_plot_dir = os.path.join(plots_dir, f"{model_name}_SHAP")
    os.makedirs(model_plot_dir, exist_ok=True)

    # SHAP summary plot (top 10 features only) using original (unscaled) data
    plt.figure()
    shap.summary_plot(shap_values, X_test_raw, feature_names=X_raw.columns, show=False, max_display=10)
    summary_path = os.path.join(model_plot_dir, f'{model_name}_SHAP_summary_top10.png')
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()
    print(f"SHAP summary plot saved at: {summary_path}")

    # Top 3 features for dependence plots (use original feature values)
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_shap)[-3:][::-1]
    top_features = [X_raw.columns[i] for i in top_features_idx]

    for feat in top_features:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_test_raw, feature_names=X_raw.columns, show=False)
        dep_path = os.path.join(model_plot_dir, f'{model_name}_SHAP_dependence_{feat}.png')
        plt.savefig(dep_path)
        plt.close()
        print(f"SHAP dependence plot saved for {feat} at: {dep_path}")
