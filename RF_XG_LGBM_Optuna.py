import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import torch
import optuna

# GPU check
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'Final_Dataframe.csv')
results_dir = os.path.join(base_dir, "Comparative_Results")

# Load & clean data
df = pd.read_csv(data_path)
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Feature sets
feature_sets = [
    ['GPR'], ['AMEX'], ['INV'], ['OVX'], ['MSCI'], [],
    ['GPR', 'AMEX'], ['GPR', 'INV'], ['AMEX', 'OVX'], ['GPR', 'MSCI'],
    ['AMEX', 'MSCI'], ['INV', 'MSCI'], ['OVX', 'MSCI'],
    ['GPR', 'AMEX', 'INV'], ['GPR', 'AMEX', 'INV', 'OVX'],
    ['GPR', 'AMEX', 'MSCI'], ['AMEX', 'OVX', 'MSCI'],
    ['GPR', 'AMEX', 'INV', 'OVX', 'MSCI']
]

def directional_accuracy(y_true, y_pred):
    return np.mean((np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])))

def create_lag_features(df, selected_features, sequence_length=30):
    features = ['WTI'] + selected_features if selected_features else ['WTI']
    df_feat = df[['Date'] + features].copy()
    lagged_data = [df_feat[features].shift(i).add_suffix(f'_lag{i}') for i in range(1, sequence_length+1)]
    df_lagged = pd.concat([df_feat] + lagged_data, axis=1).dropna()
    return df_lagged.drop(columns=['WTI', 'Date']), df_lagged['WTI'], df_lagged['Date']

def optuna_tune(model_name, X_train, y_train):
    def objective(trial):
        if model_name == "RF":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            model = RandomForestRegressor(**params, random_state=42)
        elif model_name == "XGB":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'tree_method': 'gpu_hist' if gpu_available else 'hist',
                'predictor': 'gpu_predictor' if gpu_available else 'cpu_predictor'
            }
            model = XGBRegressor(**params, random_state=42, verbosity=0)
        else:  # LGBM
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'force_col_wise': True
            }
            model = LGBMRegressor(**params, random_state=42, verbose=-1)

        model.fit(X_train, y_train)
        pred = model.predict(X_train)
        return mean_squared_error(y_train, pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    return study.best_params

def run_pipeline(model_name, features, df):
    X, y, dates = create_lag_features(df, features)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = dates[split:]

    best_params = optuna_tune(model_name, X_train, y_train)

    if model_name == "RF":
        model = RandomForestRegressor(**best_params, random_state=42)
    elif model_name == "XGB":
        model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    else:
        model = LGBMRegressor(**best_params, random_state=42, verbose=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    dir_acc = directional_accuracy(y_test.values, y_pred)

    return rmse, mae, mape, dir_acc, y_test.values, y_pred, test_dates

# Run experiments
results = []
predictions = {}
models = ['RF', 'XGB', 'LGBM']

for model_name in models:
    print(f"\n🔍 Tuning and running: {model_name}")
    for features in tqdm(feature_sets, desc=model_name):
        feature_label = '+'.join(features) if features else 'WTI_Only'
        label = f"{model_name}_{feature_label}"
        rmse, mae, mape, dir_acc, y_true, y_pred, dates = run_pipeline(model_name, features, df)
        results.append((model_name, feature_label, rmse, mae, mape, dir_acc))
        predictions[label] = (y_true, y_pred, dates)

# Save results
results_df = pd.DataFrame(results, columns=["Model", "Feature Set", "RMSE", "MAE", "MAPE", "Directional Accuracy"])
os.makedirs(results_dir, exist_ok=True)

model_dirs = {}
for model_name in models:
    model_folder = os.path.join(results_dir, model_name)
    os.makedirs(model_folder, exist_ok=True)
    model_dirs[model_name] = model_folder

    model_df = results_df[results_df["Model"] == model_name]
    model_df.sort_values("RMSE", inplace=True)
    model_df.to_csv(os.path.join(model_folder, f"{model_name}_results.csv"), index=False)
    print(f"📊 {model_name} results saved.")

# Plot predictions
for label, (y_true, y_pred, dates) in predictions.items():
    plt.figure(figsize=(10, 4))
    plt.plot(dates, y_true, label='Actual', color='black')
    plt.plot(dates, y_pred, label='Predicted', linestyle='--', color='orange')
    plt.title(f'WTI Prediction - {label}')
    plt.xlabel('Date')
    plt.ylabel('WTI Price')
    plt.legend()
    plt.tight_layout()

    model_name = label.split('_')[0]
    fname = f"{label.replace('+', '_')}.png"
    plt.savefig(os.path.join(model_dirs[model_name], fname))
    plt.close()

print("\n✅ All results and plots saved!")
for name, path in model_dirs.items():
    print(f"- {name} → {path}")
