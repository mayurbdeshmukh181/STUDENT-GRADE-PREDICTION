# Regression Model Implementation and Optimization

## Overview
This project implements a variety of regression models, both linear and ensemble-based, to predict numerical outcomes. The models are optimized using **Grid Search** and **Randomized Grid Search** techniques to find the best hyperparameters for each model.

---

## Contents
1. [Linear Models](#linear-models)
2. [Ensemble Models](#ensemble-models)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
   - Grid Search
   - Randomized Grid Search
4. [How to Use](#how-to-use)
5. [Dependencies](#dependencies)

---

## Linear Models
The following linear regression models are included:

### **1. LinearRegression**
A basic linear regression model that minimizes the mean squared error.

### **2. Ridge**
Ridge regression adds L2 regularization to the loss function to prevent overfitting.

### **3. SGDRegressor**
Stochastic Gradient Descent is used to minimize the loss function iteratively.

### **4. Lasso**
Lasso regression adds L1 regularization to the loss function, which performs feature selection by driving some coefficients to zero.

### **5. ElasticNetCV**
Combines L1 and L2 regularization with cross-validation to find the best combination of hyperparameters.

### **6. HuberRegressor**
A robust regression method that minimizes the influence of outliers on the model.

### **7. QuantileRegressor**
Predicts a specific quantile (e.g., median) instead of the mean of the response variable.

### **8. RANSACRegressor**
A robust regression model that iteratively fits the model to subsets of the data and identifies inliers.

### **9. PoissonRegressor**
Used for count data regression assuming the response variable follows a Poisson distribution.

### **10. TweedieRegressor**
Handles distributions from the Tweedie family, including compound Poisson-gamma.

### **11. GammaRegressor**
Models data that follow a gamma distribution, useful for positively skewed data.

---

## Ensemble Models
The following ensemble-based regression models are included:

### **1. DecisionTreeRegressor**
A non-linear model that uses a tree structure to split data based on feature values.

### **2. RandomForestRegressor**
An ensemble of decision trees where each tree is trained on a random subset of the data.

### **3. GradientBoostingRegressor**
An iterative method that combines weak learners (trees) to minimize the loss function.

### **4. XGBoost**
An optimized gradient boosting implementation designed for speed and performance.

### **5. LightGBM**
A gradient boosting framework optimized for efficiency, capable of handling large datasets.

### **6. CatBoost**
A gradient boosting algorithm that handles categorical features natively.

---

## Hyperparameter Optimization

### **Grid Search**
This exhaustive search method evaluates all possible combinations of hyperparameter values specified in the search space.

#### Example:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {
    'alpha': [0.1, 0.5, 1.0],
    'max_iter': [1000, 2000, 3000]
}
grid_search = GridSearchCV(estimator=Ridge(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

### **Randomized Grid Search**
This method evaluates a random subset of the hyperparameter search space, making it faster for large datasets and complex models.

#### Example:
```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'learning_rate': [0.01, 0.1, 0.2]
}
random_search = RandomizedSearchCV(estimator=GradientBoostingRegressor(), param_distributions=param_dist, scoring='r2', n_iter=10, cv=5, random_state=42)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

---

## How to Use
1. **Import Models**:
   The models are defined in dictionaries (`regressors` and `ensemble_models`). Use the key to access a specific model.
   ```python
   from sklearn.linear_model import LinearRegression

   model = regressors['LinearRegression']
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   ```

2. **Perform Optimization**:
   Use either `GridSearchCV` or `RandomizedSearchCV` to optimize hyperparameters.

3. **Evaluate Models**:
   Evaluate the model using metrics such as **Mean Squared Error (MSE)**, **R2 Score**, or **Mean Absolute Error (MAE)**.
   ```python
   from sklearn.metrics import mean_squared_error, r2_score

   mse = mean_squared_error(y_test, predictions)
   r2 = r2_score(y_test, predictions)
   print(f"MSE: {mse}, R2: {r2}")
   ```

---

## Dependencies
- Python 3.7+
- scikit-learn
- xgboost
- lightgbm
- catboost

Install required libraries:
```bash
pip install scikit-learn xgboost lightgbm catboost
```
