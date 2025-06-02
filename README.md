# WTI Oil Price Prediction using Machine Learning

## Project Overview

Created for my thesis, this repository contains project files on predicting West Texas Intermediate (WTI) crude oil prices. The project implements multiple models including Random Forest, XGBoost, LightGBM, LSTM, and GRU neural networks to forecast oil prices based on historical data and related economic indicators.

The analysis incorporates several key financial and economic variables that influence oil prices, such as the AMEX Oil Index, OVX (Oil Volatility Index), inventory levels, geopolitical risk indicators, and MSCI World Index data. Through comparative analysis of different models and feature importance evaluation using SHAP (SHapley Additive exPlanations), this project provides insights into the factors driving oil price movements and the predictive performance of various machine learning approaches.

## Repository Structure

The repository is organized as follows:

### Data Files

The `Data` directory contains all input datasets used for model training and evaluation:

- `WTI.csv`: Historical West Texas Intermediate crude oil price data
- `AMEX.csv`: AMEX Oil Index data representing oil company stock performance
- `OVX.csv`: Oil price volatility index data
- `INV.csv`: Oil inventory level data
- `GPR.csv`: Geopolitical risk index data
- `MSCI.csv`: MSCI World Index data for global market performance
- `Linechart.png`: Visualization of key data series

### Python Scripts

- `DataFrame.py`: Data preprocessing script that loads, cleans, and merges all data sources into a unified dataset for model training
- `RF_XG_LGBM.py`: Implementation of tree-based models (Random Forest, XGBoost, and LightGBM) for oil price prediction
- `RF_XG_LGBM_Optuna.py`: Hyperparameter optimization for tree-based models using Optuna framework
- `LSTM Final.py`: Long Short-Term Memory neural network implementation for time series forecasting
- `GRU Final.py`: Gated Recurrent Unit neural network implementation for time series forecasting
- `Regression_Analysis.py`: Statistical regression analysis of oil price determinants
- `SHAP_Analysis.py`: Feature importance analysis using SHAP values
- `Linechart.py`: Script for generating data visualizations

### Results and Analysis

- `Comparative_Results`: Directory containing performance metrics and visualizations for different models across multiple runs
- `SHAP_results`: Directory with SHAP analysis visualizations showing feature importance for each model
- `Consolidated_Results.xlsx`: Spreadsheet with consolidated performance metrics across all models
- `Regression_Results.csv`: Results from statistical regression analysis
- `Final_DataFrame.csv`: Processed dataset used for model training and evaluation

## Dependencies

This project requires the following Python libraries:

```
pandas
numpy
scikit-learn
xgboost
lightgbm
tensorflow/keras
matplotlib
seaborn
shap
torch (optional for GPU acceleration)
tqdm
```

## Usage Instructions

### Data Preparation

1. Clone this repository to your local machine
2. Update the data paths in `DataFrame.py` to point to your local `Data` directory
3. Run `DataFrame.py` to generate the processed dataset:

```python
python DataFrame.py
```

This will create `Final_DataFrame.csv` which is used by all model scripts

### Model Training and Evaluation

#### Tree-based Models

Run the following script to train and evaluate Random Forest, XGBoost, and LightGBM models:

```python
python RF_XG_LGBM.py
```

For hyperparameter optimization, use:

```python
python RF_XG_LGBM_Optuna.py
```

#### Neural Network Models

For LSTM neural network implementation:

```python
python "LSTM Final.py"
```

For GRU neural network implementation:

```python
python "GRU Final.py"
```

### Feature Importance Analysis

To generate SHAP analysis visualizations:

```python
python SHAP_Analysis.py
```

### Regression Analysis

For statistical regression analysis:

```python
python Regression_Analysis.py
```

## Model Performance

The repository includes comprehensive performance evaluations for all implemented models. Results are stored in the `Comparative_Results` directory, with consolidated metrics available in `Consolidated_Results.xlsx`. The analysis includes standard regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared values.

Based on the included results, the tree-based ensemble methods (particularly XGBoost and LightGBM) generally demonstrate strong predictive performance for WTI oil prices, with neural network approaches providing competitive results for certain forecasting horizons.

## Feature Importance

The SHAP analysis results in the `SHAP_results` directory provide insights into which features most significantly influence the model predictions. This analysis helps identify the key drivers of oil price movements according to each model. Common influential features across models include inventory levels, oil price volatility (OVX), and oil company stock performance (AMEX).

## Notes on File Paths

The scripts in this repository use relative and absolute file paths that may need to be adjusted based on your local environment. Before running any scripts, please review and update the following path variables as needed:

- In `DataFrame.py`: Update `data_path` variable
- In `RF_XG_LGBM.py` and other model scripts: Update `base_dir` and `data_path` if necessary

## Contributing

Contributions to improve the models, add new features, or enhance documentation are welcome. Please feel free to submit pull requests or open issues to discuss potential improvements.

## License

This project is available for academic and research purposes. Please cite this repository if you use any part of this work in your research.
