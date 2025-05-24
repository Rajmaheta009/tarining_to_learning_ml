import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load data
housing = pd.read_csv("Housing.csv")

# Split features and label
train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)

train_label = train_data["price"].copy()
test_label = test_data["price"].copy()

train_data = train_data.drop("price", axis=1)
test_data = test_data.drop("price", axis=1)

# Separate column types
num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = train_data.select_dtypes(include=['object']).columns

# Define transformers
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Transform data
train_prepared = full_pipeline.fit_transform(train_data)
test_prepared = full_pipeline.transform(test_data)

# Choose model
model = LinearRegression()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()

# Fit model
model.fit(train_prepared, train_label)

# Evaluate model
def evaluate_model(model, data, labels):
    predictions = model.predict(data)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    return rmse

def score(model, data, labels):
    scores = cross_val_score(model, data, labels, scoring="neg_mean_squared_error", cv=10)
    return np.sqrt(-scores)

# Scores
print("Training RMSE:", evaluate_model(model, train_prepared, train_label))
cv_scores = score(model, test_prepared,test_label )
print("Cross-validated RMSE scores:", cv_scores)
print("Mean CV RMSE:", cv_scores.mean())
print("Standard deviation of CV RMSE:", cv_scores.std())
