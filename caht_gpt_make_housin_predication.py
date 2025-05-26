# ✅ House Price Prediction with Ridge Regression
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL.JpegImagePlugin import jpeg_factory

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# 🔹 1. Load Dataset
# Load the dataset
data = pd.read_csv("csv_database/Housing.csv")

# Drop the 'id' column if it exists
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Define the target and features
target = 'price'
y = data[target]
X = data.drop(columns=[target])

# 🔹 2. Detect Numerical and Categorical Columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 🔹 3. Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# 🔹 4. Full Ridge Pipeline
ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

# 🔹 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 6. Grid Search for Alpha
params = {
    'regressor__alpha': [0.01, 0.1, 1, 10, 100]
}
grid = GridSearchCV(ridge_model, param_grid=params, cv=5, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)

# 🔹 7. Evaluation
print("✅ Best Parameters:", grid.best_params_)
print(f"✅ Best CV RMSE: {-grid.best_score_:.2f}")

y_pred = grid.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print(f"📊 Test RMSE: {rmse:.2f}")

# 🔹 8. Actual vs Predicted Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.6)
# plt.xlabel("Actual Prices")
# plt.ylabel("Predicted Prices")
# plt.title("Actual vs Predicted House Prices")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

joblib.dump(grid,"pkl files/house_price_prediction.pkl")