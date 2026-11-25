# Predicting O-Level Mathematics Scores

An end-to-end machine learning pipeline that predicts a student's O-level mathematics score (`final_test`) from their background, study habits, sleep patterns and school environment. The aim is to help schools identify students who may need extra support before the exam.

This is a **regression** problem: the target is a continuous score from 0 to 100.

## Folder Structure

```
├── data/
│   └── regression_bonus_practice_data.csv
├── src/
│   ├── __init__.py
│   ├── config.yaml            # file paths, features, split sizes, hyperparameter grid
│   ├── data_preparation.py    # cleaning, feature engineering, imputation, preprocessing
│   └── model_training.py      # splitting, training, tuning and evaluation
├── main.py                    # runs the full pipeline
├── requirements.txt
└── README.md
```

## Prerequisites and Installation

- Python 3.11
- conda (or any virtual environment manager)

```bash
# Create and activate an environment
conda create -n aiapenv python=3.11 -y
conda activate aiapenv

# Install dependencies
pip install -r requirements.txt
```

Place the dataset at `data/regression_bonus_practice_data.csv`.

## Instructions for Executing the Pipeline

From the project root:

```bash
python main.py
```

The pipeline logs each stage, the validation metrics for every model, a comparison table, the best model, and its final test-set metrics.

### Configuring the pipeline

All settings live in `src/config.yaml`, so you can change them without editing code:

| Parameter | Description |
|---|---|
| `file_path` | Path to the dataset |
| `target_column` | Column to predict (`final_test`) |
| `val_test_size` | Share of data held out for validation and test combined (default 0.2) |
| `val_size` | Share of the held-out data used for testing (default 0.5, giving an 80/10/10 split) |
| `param_grid` | Regularisation strengths (`alpha`) to search for Ridge and Lasso |
| `cv` | Number of cross-validation folds for tuning |
| `scoring` | Metric used to select hyperparameters |
| `numerical_features` / `nominal_features` | Features passed to the model and how each is preprocessed |

## Description of Logical Steps/Flow of the Pipeline

```
Load data → Clean data → Feature engineering → Split (train / val / test)
   → Impute attendance (train medians) → Preprocess (scale + one-hot encode)
   → Train baseline models → Tune Ridge & Lasso (GridSearchCV)
   → Compare on validation set → Evaluate best model once on test set
```

1. **Load:** read the CSV and `config.yaml` (`main.py`).
2. **Clean** (`DataPreparation.clean_data`): drop rows with a missing target, remove duplicate students, standardise inconsistent labels, and fix invalid ages.
3. **Feature engineering:** create `sleep_duration` and `class_size`, then drop identifiers and raw or helper columns.
4. **Split** (`ModelTraining.split_data`): 80% train, 10% validation, 10% test, with a fixed random seed for reproducibility.
5. **Impute** (`DataPreparation.impute_attendance`): fill missing `attendance_rate` using medians learned from the **training set only**, so no information leaks from the validation or test sets.
6. **Preprocess:** a `ColumnTransformer` scales numerical features and one-hot encodes categorical features. It sits inside each model's `Pipeline`, so it is fitted on training data only.
7. **Train baselines:** Linear Regression, Ridge and Lasso with default settings.
8. **Tune:** grid search with 5-fold cross-validation over `alpha` for Ridge and Lasso.
9. **Select and evaluate:** pick the model with the best validation R², then evaluate it **once** on the untouched test set.

## Overview of Key Findings from EDA

The findings below informed every cleaning, feature engineering and modelling decision in the pipeline.

**Data quality**
- 15,900 raw records. 495 rows (about 3%) had no `final_test` score and were dropped.
- **846 duplicate student records.** They were identical apart from `bag_color` (random) and, in 83 cases, a missing `attendance_rate`. Of these, 138 were exact copies, 625 differed only in `bag_color`, and 83 also differed in `attendance_rate`.
- Inconsistent labels: `CCA` had mixed casing (`Clubs`/`CLUBS`) and `tuition` mixed `Yes`/`Y` and `No`/`N`.
- `age` contained invalid values (6, 5, -5, -4) alongside the expected 15 and 16. The roughly equal counts of 5/6 mirrored those of 15/16, which suggests a dropped leading "1".
- After cleaning, there are **14,558 unique students**.

**Relationships with `final_test`** (Pearson correlation)

| Feature | r | Interpretation |
|---|---|---|
| `class_size` | −0.50 | Strongest predictor: larger classes score lower. The relationship is **non-linear**, with distinct low-scoring clusters at some sizes. |
| `number_of_siblings` | −0.36 | More siblings, lower scores |
| `attendance_rate` | +0.35 | Higher attendance, higher scores |
| `sleep_duration` | +0.33 | Less sleep, lower scores |
| `hours_per_week` | −0.15 | Weak negative effect, possibly because struggling students study more |
| `age` | 0.00 | No relationship |

**Sleep and attendance are strongly linked (r = 0.89).** Each hour of lost sleep corresponds to roughly 10 points lower attendance: median attendance is about 95% at 8 hours of sleep and about 50% at 4 hours. About 92% of students sleep 8 hours, and the shortfall comes from students who sleep after midnight.

**Categorical features** (mean `final_test` per group)
- `tuition`: Yes 70.5 vs No 62.9
- `learning_style`: Visual 71.6 vs Auditory 63.9
- `direct_admission`: Yes 72.5 vs No 65.0
- `CCA`: students with no CCA ("none", 76.8) or no recorded CCA ("unknown", 76.7) score about 12 points higher than those in arts, clubs or sports (about 64). The near-identical means suggest most missing CCA values are students without a CCA.
- `gender` (67.0 vs 67.3) and `mode_of_transport` (67.1 to 67.3) show no meaningful difference.

## Feature Handling Description

| Feature | Source | Handling |
|---|---|---|
| `number_of_siblings` | Original | Standard scaling |
| `hours_per_week` | Original | Standard scaling |
| `attendance_rate` | Original | Missing values (about 5%) imputed with the training-set median **per `sleep_duration` group**, then standard scaling |
| `sleep_duration` | **Engineered:** wake time minus sleep time, in hours. Sleep times after midnight are shifted by +24h. | Standard scaling |
| `class_size` | **Engineered:** `n_male + n_female` | Standard scaling |
| `direct_admission` | Original | One-hot encoding (binary, so a single 0/1 column) |
| `CCA` | Original | Lowercased. Missing values filled as `"unknown"`, kept separate from `"none"`. One-hot encoded. |
| `learning_style` | Original | One-hot encoding (binary) |
| `gender` | Original | One-hot encoding (binary) |
| `tuition` | Original | `Y`/`N` standardised to `Yes`/`No`, then one-hot encoded (binary) |
| `mode_of_transport` | Original | One-hot encoding |
| `age` | Original | Negative values made positive, 5 and 6 corrected to 15 and 16, the single age-4 record dropped. One-hot encoded. |

**Dropped features**

| Feature | Reason |
|---|---|
| `index` | Row number with no meaning |
| `student_id` | Identifier |
| `bag_color` | Random, not predictive. It was also the main source of near-duplicate records. |
| `sleep_time`, `wake_time` | Raw strings, replaced by `sleep_duration` |
| `n_male`, `n_female` | Combined into `class_size`. Keeping all three would cause perfect multicollinearity. |

One-hot encoding uses `drop="if_binary"` to avoid the dummy-variable trap for two-category features, and `handle_unknown="ignore"` so unseen categories at prediction time do not cause errors.

## Explanation of Model Choices

The project uses **linear regression models**, consistent with the scope of the AISG Foundations linear regression module.

| Model | Why it was included |
|---|---|
| **Linear Regression** | Interpretable baseline. Each coefficient shows how much a feature moves the predicted score. |
| **Ridge (L2)** | Shrinks coefficients to reduce overfitting and handles correlated features such as `attendance_rate` and `sleep_duration` (r = 0.89) more stably. |
| **Lasso (L1)** | Can shrink weak coefficients to exactly zero, acting as automatic feature selection. |

Ridge and Lasso are tuned with `GridSearchCV` (5-fold cross-validation) over `alpha` values from 0.001 to 1000. All numerical features are standardised so that regularisation penalises every feature on the same scale.

## Evaluation of Models

**Metrics**
- **MAE:** the average error in marks. This is the easiest to explain to teachers.
- **RMSE:** penalises large errors more heavily.
- **R²:** the share of variation in scores the model explains. Used to select the best model.

**Validation results**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 7.19 | 8.97 | 0.596 |
| Ridge (alpha = 1) | 7.19 | 8.97 | 0.596 |
| Ridge (tuned) | 7.19 | 8.98 | 0.596 |
| Lasso (tuned) | 7.19 | 8.98 | 0.596 |
| Lasso (alpha = 1) | 7.88 | 9.78 | 0.521 |

**Final test result (Linear Regression):** MAE **6.99**, RMSE **8.75**, R² **0.593**

**Interpretation**
- Predictions are off by about **7 marks** on average, and the model explains about **59%** of the variation in scores.
- Test performance closely matches validation performance, so the model generalises well.
- Regularisation gave no improvement, which means the model is **not overfitting**. Its limit comes from being linear.
- Lasso with alpha = 1 over-penalised and dropped useful features. Tuning brought it back to about 0.596.
- The EDA showed a non-linear class-size effect and possible interactions between features, which a linear model cannot fully capture.

**Future work:** tree-based models such as Random Forest or Gradient Boosting could capture these non-linear effects and interactions. Polynomial or binned `class_size` features could also be tested within the linear framework.

## Considerations for Deployment

- **Interpretability:** teachers and parents need to understand why a student is flagged. Linear coefficients can be explained directly (for example, "each additional sibling is associated with about X fewer marks"), which supports trust and action.
- **Fairness:** `gender` showed no effect on scores and could be excluded to avoid any perception of bias. Error rates should be checked across groups (gender, transport mode, CCA) so that no group is systematically under- or over-predicted.
- **Use as support, not judgement:** predictions should flag students who may need help, not label them. An average error of about 7 marks should be communicated alongside every prediction.
- **Data quality at input:** the cleaning rules (label standardisation, age corrections, duplicate removal) must run on new data too. Input validation should reject or flag out-of-range values such as invalid ages or attendance above 100%.
- **Data drift:** student behaviour, class sizes and syllabus changes can shift over time. Monitor input feature distributions and prediction error against actual exam results each year, and retrain when performance degrades.
- **Reproducibility:** configuration in `config.yaml`, fixed random seeds and pinned dependencies in `requirements.txt` allow the pipeline to be rerun with identical results.
- **Privacy:** student data is sensitive. Identifiers such as `student_id` are dropped before modelling, and any deployment should follow data protection requirements (PDPA in Singapore).
