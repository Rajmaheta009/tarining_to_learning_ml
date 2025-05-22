import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge  # Added Ridge for regularization

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor

# Load dataset
housing_data = pd.read_csv("Housing.csv")

# -------------------- Outlier Removal --------------------
# Remove outliers from target variable 'price' using IQR method
Q1 = housing_data['price'].quantile(0.25)
Q3 = housing_data['price'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter data within bounds (removes extreme outliers)
housing_data_filtered = housing_data[(housing_data['price'] >= lower_bound) & (housing_data['price'] <= upper_bound)]

# -------------------- Feature Selection --------------------
# Calculate correlation matrix only for numeric features to avoid conversion error
corr_matrix = housing_data_filtered.select_dtypes(include=[np.number]).corr()

# Select features with absolute correlation > 0.1 with price
selected_features = corr_matrix['price'][abs(corr_matrix['price']) > 0.1].index.tolist()

# Remove 'price' from features list
if 'price' in selected_features:
    selected_features.remove('price')

# Keep only selected features + price
housing_data_selected = housing_data_filtered[selected_features + ['price']]


# -------------------- Split data --------------------
def test_train_dataset(housing):
    train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)
    return train_data, test_data

train_all_data, test_all_data = test_train_dataset(housing_data_selected)

train_data = train_all_data.drop("price", axis=1)
train_data_label = train_all_data['price'].copy()

test_data = test_all_data.drop("price", axis=1)
test_data_label = test_all_data['price'].copy()

# -------------------- Preprocessing Pipeline --------------------
def make_pipeline(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

pip = make_pipeline(train_data)

# -------------------- Model & Hyperparameter Tuning --------------------
# Using Ridge regression with alpha hyperparameter tuning for regularization to reduce overfitting

ridge = Ridge()

full_pipeline = Pipeline(steps=[
    ('preprocessor', pip),
    ('regressor', ridge)
])

# Define hyperparameter grid for Ridge alpha
param_grid = {'regressor__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(train_data, train_data_label)

print("Best alpha for Ridge:", grid_search.best_params_)

# Use best estimator after tuning
best_model = grid_search.best_estimator_

# -------------------- Evaluation Functions --------------------
def evaluate_model(model):
    housing_prediction = model.predict(train_data)
    mse = mean_squared_error(train_data_label, housing_prediction)
    rmse = np.sqrt(mse)
    return rmse

def score(model):
    scores = cross_val_score(model, test_data, test_data_label,
                             scoring="neg_mean_squared_error", cv=10)
    score_result = np.sort(-scores)
    return score_result

print("MY Score is :--- ", score(best_model))
print("Mean : --", score(best_model).mean())
print("std : --", score(best_model).std())
print("Train RMSE:", evaluate_model(best_model))
mean_mse = score(best_model).mean()
C_RMSE = np.sqrt(mean_mse)
print("Cross-validated RMSE:", C_RMSE)

# print(housing_data['price'].describe())
