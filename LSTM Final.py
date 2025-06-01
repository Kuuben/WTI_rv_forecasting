import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Define base directory as the location of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load and clean data
data_path = os.path.join(base_dir, 'Final_Dataframe.csv')
df = pd.read_csv(data_path)
df = df.dropna()

# Ensure 'Date' is datetime with day-first format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

feature_sets = [
    ['GPR'], ['AMEX'], ['INV'], ['OVX'], ['MSCI'], [],
    ['GPR', 'AMEX'], ['GPR', 'INV'], ['AMEX', 'OVX'], ['GPR', 'MSCI'],
    ['AMEX', 'MSCI'], ['INV', 'MSCI'], ['OVX', 'MSCI'],
    ['GPR', 'AMEX', 'INV'], ['GPR', 'AMEX', 'INV', 'OVX'],
    ['GPR', 'AMEX', 'MSCI'], ['AMEX', 'OVX', 'MSCI'],
    ['GPR', 'AMEX', 'INV', 'OVX', 'MSCI']
]

# Directional Accuracy Function
def directional_accuracy(y_true, y_pred):
    return np.mean((np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])))

# LSTM pipeline
def run_lstm_pipeline(selected_features, df, sequence_length=30):
    features = ['WTI'] + selected_features if selected_features else ['WTI']
    data = df[['Date'] + features].copy()
    data.set_index('Date', inplace=True)

    # Split before sequence creation
    split_index = int(0.8 * len(data))
    train_df = data.iloc[:split_index]
    test_df = data.iloc[split_index:]

    # Scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    def create_sequences(df_slice, fit_scaler=False):
        X_seq, y_seq, date_seq = [], [], []
        data_vals = df_slice.values
        dates = df_slice.index

        if fit_scaler:
            feature_scaler.fit(data_vals)

        scaled_vals = feature_scaler.transform(data_vals)

        for i in range(sequence_length, len(scaled_vals)):
            X_seq.append(scaled_vals[i-sequence_length:i])
            y_seq.append(data_vals[i][0])  # Close price is always at index 0
            date_seq.append(dates[i])

        return np.array(X_seq), np.array(y_seq).reshape(-1, 1), date_seq

    X_train, y_train_raw, _ = create_sequences(train_df, fit_scaler=True)
    X_test, y_test_raw, test_dates = create_sequences(test_df, fit_scaler=False)

    y_train = target_scaler.fit_transform(y_train_raw)
    y_test = target_scaler.transform(y_test_raw)

    # Build model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, 
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_actual = target_scaler.inverse_transform(y_test)

    rmse = np.sqrt(np.mean((y_pred.flatten() - y_actual.flatten()) ** 2))
    mae = np.mean(np.abs(y_pred.flatten() - y_actual.flatten()))
    mape = np.mean(np.abs((y_pred.flatten() - y_actual.flatten()) / y_actual.flatten())) * 100
    dir_acc = directional_accuracy(y_actual.flatten(), y_pred.flatten())

    return rmse, mae, mape, dir_acc, y_actual.flatten(), y_pred.flatten(), test_dates

# Run experiments
results = []
all_predictions = {}

for features in feature_sets:
    label = '+'.join(features) if features else 'WTI_Only'
    print(f"Testing feature set: {label}")
    rmse, mae, mape, dir_acc, y_actual, y_pred, test_dates = run_lstm_pipeline(features, df)
    results.append((label, rmse, mae, mape, dir_acc))
    all_predictions[label] = (y_actual, y_pred, test_dates)

# Save results to CSV
results_dir = os.path.join(base_dir, "Results")
os.makedirs(results_dir, exist_ok=True)

results_df = pd.DataFrame(results, columns=["Feature Set", "RMSE", "MAE", "MAPE", "Directional Accuracy"])
output_csv_path = os.path.join(results_dir, "lstm_results.csv")
results_df.sort_values("RMSE").to_csv(output_csv_path, index=False)
print(f"\nResults saved to: {output_csv_path}")

# Save plots
plots_dir = os.path.join(results_dir, "LSTM_Plots")
os.makedirs(plots_dir, exist_ok=True)

for label, (y_actual, y_pred, test_dates) in all_predictions.items():
    plt.figure(figsize=(10, 4))
    plt.plot(test_dates, y_actual, label='Actual', color='black')
    plt.plot(test_dates, y_pred, label='Predicted', linestyle='--', color='orange')
    plt.title(f'WTI Prediction - Features: {label}')
    plt.xlabel('Date')
    plt.ylabel('WTI Price')
    plt.legend()
    plt.tight_layout()
    image_filename = f"{label.replace('+', '_')}.png"
    image_path = os.path.join(plots_dir, image_filename)
    plt.savefig(image_path)
    plt.close()

print(f"Plots saved in: {plots_dir}")
