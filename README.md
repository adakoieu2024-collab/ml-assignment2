# ML Assignment II: Bike Sharing Demand Prediction

**Author:** Alesia Dako  
**Course:** AI Machine Learning Foundations  
**Date:** May 2026


## Overview

This repository contains the second individual assignment for the 
AI Machine Learning Foundations course. The objective is to predict 
hourly bike rental demand for the Capital Bikeshare system in 
Washington D.C. using three supervised learning models of increasing 
complexity: Linear Regression, Random Forest, and Gradient Boosting.


## Repository Structure

ml-assignment2/

├── assignment_2_Alesia_Dako.ipynb   ← main notebook

├── data/

│   └── hour.csv                     ← UCI Bike Sharing Dataset (hourly)

├── README.md

└── .gitignore


## Dataset

The dataset is the **UCI Bike Sharing Dataset** (hourly version), 
available at:  
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Download the zip file, extract `hour.csv`, and place it in the 
`data/` folder before running the notebook. The dataset contains 
17,379 hourly records from the Capital Bikeshare system covering 
2011 and 2012, with weather, seasonal, and temporal features.


## Requirements

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-optimize xgboost
```

Or run the first cell of the notebook, which installs all packages 
automatically:

```python
%pip install numpy pandas matplotlib seaborn scikit-learn scikit-optimize xgboost
```


## How to Run

1. Clone the repository:
```bash
git clone https://github.com/adakoieu2024-collab/ml-assignment2.git
cd ml-assignment2
```

2. Place `hour.csv` in the `data/` folder

3. Open the notebook:
```bash
jupyter notebook assignment_2_Alesia_Dako.ipynb
```

4. Run all cells in order:  
   **Kernel → Restart & Run All**


## Notebook Structure

| Task | Description |
|------|-------------|
| Task 1 | Exploratory Data Analysis (EDA) |
| Task 2 | Chronological Data Splitting (60/20/20) |
| Task 3 | Feature Engineering |
| Task 4 | Baseline Model: Linear Regression |
| Task 5 | Random Forest Regressor |
| Task 6 | Gradient Boosting Regressor |
| Task 7 | Hyperparameter Tuning (RandomizedSearchCV + BayesSearchCV) |
| Task 8 | Iterative Evaluation and Refinement |
| Task 9 | Final Model Selection and Test Evaluation |


## Key Modeling Decisions

- **Chronological split:** (60/20/20) no shuffle, to prevent temporal 
  leakage and simulate real-world forecasting
- **All preprocessing fit on training set only:** scalers, encoders, 
  and imputers are fit on training data and applied to validation and 
  test sets
- **Cyclical encoding:** for `hr` and `weekday` sine/cosine transforms 
  preserve circular adjacency
- **`atemp` dropped:** near perfect correlation with `temp` (r≈0.99)
- **Interaction term `temp × hum`:** motivated by grouped correlation 
  analysis in EDA
- **Test set used exactly once:** in Task 9 only, after all model 
  selection decisions were made on the validation set


## Evaluation Metrics

All models are evaluated using:
- **MSE:** Mean Squared Error
- **MAE:** Mean Absolute Error  
- **R²:** Coefficient of Determination

Metrics are reported on both training and validation sets throughout 
Tasks 4–8 to monitor the bias/variance tradeoff. The test set is 
evaluated once in Task 9.


## Note for Grader

The notebook loads the dataset using a relative path:
```python
pd.read_csv("data/hour.csv")
```

Please ensure `hour.csv` is placed in the `data/` folder before 
running. The notebook contains no Colab-specific functions and runs 
fully with **Kernel → Restart & Run All**.
