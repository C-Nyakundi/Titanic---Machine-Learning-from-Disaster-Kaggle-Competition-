# Titanic-Machine-Learning-from-Disaster-Kaggle-Competition-

This project is a solution to the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition on Kaggle. The objective is to build predictive models that determine whether a passenger survived or not, based on features such as age, sex, class, and ticket fare.

## Project Overview

This notebook walks through the end-to-end pipeline of building several classification models using the Titanic dataset. It covers:

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Handling missing values
- Feature transformation using pipelines
- Model training using various classifiers
- Hyperparameter tuning using `GridSearchCV`
- Final model evaluation and predictions for submission

## Models Used

The following machine learning algorithms were explored:

1. **Random Forest Classifier**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Classifier (SVC)**
5. **Logistic Regression**
6. **Naive Bayes (GaussianNB)**

Each model was tested using hyperparameter tuning, and the best configurations were recorded.

## Notable Results

- **Random Forest** and **Decision Tree** yielded the best performance during cross-validation.
- Example of best Decision Tree parameters:
  - `criterion`: `'entropy'`
  - `max_depth`: `20`
  - `min_samples_leaf`: `4`
  - `min_samples_split`: `15`
  - **Score**: `0.8119` (accuracy)

## Key Techniques

- **Pipelines** for consistent and clean preprocessing of categorical and numerical data
- **`ColumnTransformer`** for parallel feature transformation
- **`SimpleImputer`**, `OrdinalEncoder`, and `OneHotEncoder` for feature handling
- **`GridSearchCV`** for exhaustive hyperparameter tuning
- **Feature engineering** from names, tickets, and cabins

## Data Preprocessing Highlights

- Title extraction from names
- Age binning using quantiles
- Ticket frequency as a feature
- Cabin availability encoded as binary
- Combined pipelines to streamline transformation


## Evaluation

Models were evaluated using cross-validation and scored based on accuracy. Final outputs were prepared for Kaggle submission using:

```python
submission.to_csv("submission.csv", index=False)
```

##  How to Run

1. Clone the repository
2. Install dependencies:
3. Run the notebook:

## Future Improvements

- Feature selection using SHAP or permutation importance
- Ensemble stacking of top models
- Use of more advanced models like XGBoost or LightGBM
