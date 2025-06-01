import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------------------
# STEP 1: Set Folder & Load Files
# -------------------------------

data_path = r"D:\Documents\Python\MLThesis\Data"

# Load CSVs
df_wti = pd.read_csv(os.path.join(data_path, "WTI.csv"))
df_gpr = pd.read_csv(os.path.join(data_path, "GPR.csv"))
df_amex = pd.read_csv(os.path.join(data_path, "AMEX.csv"))
df_inv = pd.read_csv(os.path.join(data_path, "INV.csv"), encoding='latin1')
df_ovx = pd.read_csv(os.path.join(data_path, "OVX.csv"))
df_msci = pd.read_csv(r'D:\Documents\Python\MLThesis\WTI_ML\Data\MSCI.csv')

# Standardize column names
df_wti.columns = ['Date', 'WTI']
df_gpr.columns = ['Date', 'GPR_Index']
df_amex.columns = ['Date', 'AMEX_Value']
df_inv.columns = ['Date', 'INV_Value']
df_ovx.columns = ['Date', 'OVX_Value']
df_msci.columns = ['Date', 'MSCI']

# -------------------------------
# STEP 2: Parse Dates
# -------------------------------

for df in [df_wti, df_gpr, df_amex, df_inv, df_ovx, df_msci]:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

common_frequency = 'D'
common_start_date = '2010-01-01'
common_end_date = '2019-12-31'

df_inv = df_inv.drop_duplicates(subset='Date')
df_inv = df_inv.dropna(subset=['Date'])
df_inv = df_inv.sort_values(by='Date').reset_index(drop=True)
print("df_inv Date column:")
print(df_inv['Date'].head())
print("Is Date column monotonic increasing?", df_inv['Date'].is_monotonic_increasing)
df_inv = df_inv.set_index('Date').resample(common_frequency).ffill().reset_index()

# -------------------------------
# STEP 3: Align All Datasets
# -------------------------------

def clean_resample(df):
    return (
        df.drop_duplicates(subset='Date')
        .sort_values(by='Date')
        .set_index('Date')
        .resample(common_frequency)
        .ffill()
        .bfill()
        .reset_index()
    )

df_wti = clean_resample(df_wti)
df_gpr = clean_resample(df_gpr)
df_amex = clean_resample(df_amex)
df_inv = clean_resample(df_inv)
df_ovx = clean_resample(df_ovx)
df_msci = clean_resample(df_msci)

# Filter to date range
df_wti = df_wti[(df_wti['Date'] >= common_start_date) & (df_wti['Date'] <= common_end_date)].copy()
df_gpr = df_gpr[(df_gpr['Date'] >= common_start_date) & (df_gpr['Date'] <= common_end_date)].copy()
df_amex = df_amex[(df_amex['Date'] >= common_start_date) & (df_amex['Date'] <= common_end_date)].copy()
df_inv = df_inv[(df_inv['Date'] >= common_start_date) & (df_inv['Date'] <= common_end_date)].copy()
df_ovx = df_ovx[(df_ovx['Date'] >= common_start_date) & (df_ovx['Date'] <= common_end_date)].copy()
df_msci = df_msci[(df_msci['Date'] >= common_start_date) & (df_msci['Date'] <= common_end_date)].copy()


# -------------------------------
# STEP 4: Check duplicates BEFORE merging
# -------------------------------

def check_and_drop_duplicates(df, name):
    print(f"{name} shape before drop_duplicates: {df.shape}")
    dup_count = df['Date'].duplicated().sum()
    print(f"{name} duplicate Dates count: {dup_count}")
    if dup_count > 0:
        print(f"Dropping duplicates in {name} based on Date...")
        df = df.drop_duplicates(subset='Date').reset_index(drop=True)
        print(f"{name} shape after drop_duplicates: {df.shape}")
    # Check date range too
    print(f"{name} date range: {df['Date'].min()} to {df['Date'].max()}")
    return df

df_wti = check_and_drop_duplicates(df_wti, 'df_wti')
df_gpr = check_and_drop_duplicates(df_gpr, 'df_gpr')
df_amex = check_and_drop_duplicates(df_amex, 'df_amex')
df_inv = check_and_drop_duplicates(df_inv, 'df_inv')
df_ovx = check_and_drop_duplicates(df_ovx, 'df_ovx')
df_msci = check_and_drop_duplicates(df_msci, 'df_msci')

# -------------------------------
# STEP 5: Merge All on Date
# -------------------------------

df = df_wti.copy()
df = df.merge(df_gpr, on='Date', how='left')
df = df.merge(df_amex, on='Date', how='left')
df = df.merge(df_inv, on='Date', how='left')
df = df.merge(df_ovx, on='Date', how='left')
df = df.merge(df_msci, on='Date', how='left')

# -------------------------------
# STEP 6: Export Result
# -------------------------------

output_path = os.path.join(os.path.dirname(__file__), "Final_Dataframe.csv")
df.to_csv(output_path, index=False)
print(f"Saved final merged DataFrame to: {output_path}")

print(df.head())

