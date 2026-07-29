# Adult Census Income Classification

## Overview

This project builds and compares several classification models to predict whether an individual's annual income exceeds $50K, using the [Adult Census Income Dataset](https://www.kaggle.com/datasets/priyamchoksi/adult-census-income-dataset) from Kaggle (also known as the "Census Income" / "Adult" dataset).

The notebook walks through a complete ML workflow: data understanding, cleaning, feature engineering, model training, and performance evaluation across five algorithms.

## Dataset

- **Source:** Kaggle — `priyamchoksi/adult-census-income-dataset`
- **Target variable:** `income` (`>50K` or `<=50K`), encoded as binary (1 / 0)
- **Loading:** Downloaded automatically at runtime via the `kagglehub` library

## Workflow

### Task 1 — Dataset Understanding
Loads the CSV, inspects dimensions and structure (`df.info()`), and reviews the class distribution of the target variable.

### Task 2 — Data Cleaning
- Strips leading/trailing whitespace from all text (object) columns
- Converts hidden missing values (marked as `?`) to `NaN`
- Imputes missing values in `workclass`, `occupation`, and `native.country` using each column's mode
- Detects and drops duplicate records

### Task 3 — Feature Engineering
- Encodes `income` as binary (1 = `>50K`, 0 = `<=50K`)
- One-hot encodes categorical features (`pd.get_dummies`, drop-first to avoid multicollinearity)
- Performs a stratified 80/20 train-test split (`random_state=42`)
- Applies `StandardScaler` to normalize feature values

### Task 4 — Model Building
Trains five classification algorithms on the scaled training data:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### Task 5 — Performance Evaluation
Evaluates each model on the test set using Accuracy, Precision, Recall, F1-Score, and ROC-AUC, and compiles the results into a summary table.

## Results

| Algorithm            | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  |  0.8236  |   0.7000  | 0.4418 |  0.5417  |  0.8491 |
| Decision Tree        |  0.8087  |   0.5921  | 0.6083 |  0.6001  |  0.7395 |
| Random Forest        |  0.8551  |   0.7304  | 0.6116 |  0.6657  |  0.9037 |
| KNN                  |  0.8274  |   0.6531  | 0.5732 |  0.6105  |  0.8479 |
| SVM                  |  0.8491  |   0.7509  | 0.5394 |  0.6278  |  0.8939 |

**Random Forest** achieved the best overall performance, with the highest accuracy, F1-score, and ROC-AUC among the five models tested.

## Requirements

```
numpy
pandas
scikit-learn
kagglehub
```

Install with:
```bash
pip install numpy pandas scikit-learn kagglehub
```

## How to Run

1. Ensure you have a Kaggle account and API credentials configured (required by `kagglehub` to download the dataset).
2. Open the notebook `Mohur_Datta_23BAI11091.ipynb` in Jupyter.
3. Run all cells sequentially — the dataset will be downloaded automatically on the first code cell.

## Notes

- `random_state=42` is used throughout for reproducibility.
- The train-test split is stratified to preserve the original class balance of the `income` variable.
