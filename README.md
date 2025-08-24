
# Student Score Prediction

## Overview

This project predicts student exam scores based on study-related data using **Linear Regression**.
The goal is to model the relationship between predictor variables (e.g., hours studied) and target scores to aid in academic performance analysis.

Two different modeling approaches are demonstrated:

1. **Model 1** – Using a single feature (`Hours_Studied`) with scaling.
2. **Model 2** – Using multiple preprocessed features without scaling (already cleaned and ready).

---

## Dataset

The dataset is derived from preprocessed CSV files containing student performance information.
Two processed versions are used:

* **Version 1:** `cleaned_student_data_1.csv`
* **Version 2:** Separate train/test CSVs for `X_train`, `X_test`, `y_train`, and `y_test`.

**Key features:**

* `Hours_Studied`: Number of study hours per week.
* Other engineered or selected features (for Model 2) are sourced from the second preprocessed dataset.

**Target variable:**

* `Exam_Score`: Numerical score in the final exam.

---

## Methodology

### Model 1: Single Feature with Standard Scaling

1. **Data Splitting:**

   * 80% training, 20% testing.
2. **Feature Scaling:**

   * `StandardScaler` used to normalize `Hours_Studied`.
3. **Model Training:**

   * Simple **Linear Regression** fitted on scaled data.
4. **Evaluation:**

   * Metrics: R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
   * Visualization: Scatter plot of actual vs. predicted scores.

### Model 2: Multiple Features (Preprocessed Dataset)

1. **Preprocessed Features:**

   * Loaded directly from `X_train.csv` and `X_test.csv`.
2. **Model Training:**

   * **Linear Regression** fitted without additional scaling (already preprocessed).
3. **Evaluation:**

   * Metrics: R² Score, MAE, RMSE.
   * Visualization: Scatter plot with perfect prediction line.

---

## Evaluation Metrics

For both models, the following metrics are calculated:

* **R² Score:** Measures how well the model explains variance in the target.
* **MAE (Mean Absolute Error):** Average magnitude of errors.
* **RMSE (Root Mean Squared Error):** Square root of average squared errors, more sensitive to large errors.

These metrics are displayed along with plots comparing actual and predicted values.

---

## Visualizations

Each model produces:

* Scatter plot of **Actual vs. Predicted** exam scores.
* Red dashed line representing perfect prediction.
* Helps assess bias and variance visually.

---

## Future Improvements

1. **Feature Engineering**

   * Include additional predictors such as attendance, participation in extracurricular activities, and sleep patterns.
   * Create interaction terms (e.g., `Hours_Studied × Sleep_Hours`) to capture combined effects.

2. **Model Selection**

   * Experiment with more advanced algorithms like **Random Forests**, **Gradient Boosting (XGBoost, LightGBM)**, or **Support Vector Regression** to capture non-linear relationships.
   * Compare linear and non-linear models to determine which generalizes better.

3. **Hyperparameter Tuning**

   * Use **GridSearchCV** or **RandomizedSearchCV** to find optimal model parameters.
   * For example, regularization parameters in **Ridge** or **Lasso Regression** could help improve performance and reduce overfitting.

4. **Cross-Validation**

   * Implement **k-fold cross-validation** instead of a single train-test split to better estimate generalization performance.

5. **Error Analysis**

   * Investigate outliers where predictions deviate significantly from actual scores.
   * Identify patterns in underperformance and overperformance to refine preprocessing or feature selection.

6. **Data Expansion**

   * Collect more data from different academic years or institutions to improve model robustness.
   * Incorporate real-time data sources (e.g., online study tracker logs).

7. **Explainability & Interpretability**

   * Use **SHAP** or **LIME** to explain individual predictions and feature importance.
   * This can help educators understand why a certain prediction was made.

---

