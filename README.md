# Heart-Disease-Prediction-ML

## Overview  
This repository implements Decision Tree and Random Forest classifiers on cardiovascular health data to predict heart disease risk. It demonstrates a complete ML pipeline—from exploratory analysis to hyperparameter tuning and model interpretation.

## Objectives  
- Preprocess clinical data (handle missing values, feature scaling). 
- Train and compare Decision Tree vs. Random Forest models.  
- Evaluate using clinical metrics (accuracy, precision, recall).  
- Interpret results via feature importance and SHAP values.  

## Tools & Technologies  
**Language**: Python  
**Environment**: Google Colab  
**Libraries**:  
- `pandas`, `numpy` – Data handling  
- `matplotlib`, `seaborn` – Visualization  
- `scikit-learn` – ML modeling (DecisionTreeClassifier, RandomForestClassifier)  
- `SHAP` – Model interpretability  

## Workflow & Highlights  

### 1. Data Preprocessing  
- Checked for missing values (no imputation needed).  
- Scaled numerical features using `StandardScaler`.  
- Retained all 13 clinical features for modeling.  

### 2. Model Building  
- **Decision Tree**: Tuned `max_depth = 10` using validation.  
- **Random Forest**: Tuned with `GridSearchCV` (100 estimators, `max_depth = 10`, best params logged).  
- Stratified 70-30 train-test split.  

### 3. Evaluation Metrics  

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Decision Tree       | 97.1%    | 0.97      | 0.97   | 0.97     |
| Random Forest       | 98.0%    | 0.98      | 0.98   | 0.98     |
| Tuned Random Forest | 98.0%    | 0.98      | 0.98   | 0.98     |

### 4. Cross-Validation Results  

| Model               | Cross-Validation Accuracy |
|---------------------|---------------------------|
| Decision Tree       | 100%                      |
| Random Forest       | 99.7%                     |
| Tuned Random Forest | 99.4%                     |

### 5. Advanced Analysis  
- Feature importance analysis from both Decision Tree and Random Forest  
- SHAP visualizations for individual predictions and global insights  
- Threshold tuning based on clinical considerations  

## Visual Outputs  
- Accuracy vs. Max Depth (Decision Tree)  
- Feature Importance Plot (Tuned & Untuned Random Forest)  
- Confusion Matrices (Decision Tree and Random Forest)  
- Accuracy Comparison Bar Plot  

## Key Insights  
- Random Forest (tuned) achieved **98% accuracy** on test data.  
- Decision Tree performed nearly as well with proper depth control.  
- Top predictive features include **cp**, **oldpeak**, **thalach**, and **ca**.  
- Exercise-induced angina and ST depression significantly influence predictions.  
- SHAP analysis improved model explainability, important for healthcare settings.  

## Learning Outcomes  
- Practical ML model deployment on healthcare data.  
- Experience in hyperparameter tuning and validation.  
- Interpreting ML models for clinical decision support using SHAP.  
