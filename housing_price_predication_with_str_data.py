import joblib
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

# Sklearn modules
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load dataset
housing_data = pd.read_csv("csv_database/Housing.csv")

# Split the data
def test_train_dataset(housing):
    return train_test_split(housing, test_size=0.2, random_state=42)

train_all_data, test_all_data = test_train_dataset(housing_data)

train_data = train_all_data.drop("price", axis=1)
train_data_label = train_all_data["price"].copy()

test_data = test_all_data.drop("price", axis=1)
test_data_label = test_all_data["price"].copy()

# Define preprocessing
numeric_features = train_data.select_dtypes(include=[np.number]).columns
categorical_features = train_data.select_dtypes(exclude=[np.number]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Define model
model = RandomForestRegressor()

# Combine preprocessing and model
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Fit pipeline
full_pipeline.fit(train_data, train_data_label)

# Evaluate on sample data
# some_data = test_data.iloc[:5]
# some_label = test_data_label.iloc[:5]
# result = full_pipeline.predict(some_data)

# Evaluation
def evaluate_model():
    prediction = full_pipeline.predict(test_data)
    mse = mean_squared_error(test_data_label, prediction)
    rmse = np.sqrt(mse)
    return rmse

def score():
    scores = cross_val_score(full_pipeline, test_data, test_data_label,
                             scoring="neg_mean_squared_error", cv=3)
    return np.sort(-scores)

# Print results
cv_scores = score()
print("\nCross-validation Scores:", cv_scores)
print("Mean CV MSE:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())
print("Test RMSE:", evaluate_model())
print("Cross-Validated RMSE:", np.sqrt(cv_scores.mean()))

joblib.dump((model,full_pipeline),"pkl files/housing_price_prediction_with_str_data.pkl")