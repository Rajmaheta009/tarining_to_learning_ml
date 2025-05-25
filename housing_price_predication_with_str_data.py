# that is python in play with array nad data show
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

# that is sklearn library component and tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# that is  modal lost
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# Load dataset
housing_data = pd.read_csv("csv_database/Housing.csv")

# Function to split dataset into train and test sets
def test_train_dataset(housing):
    train_data, test_data = train_test_split(housing, test_size=0.2, random_state=42)
    return train_data, test_data

# Split the housing data
train_all_data, test_all_data = test_train_dataset(housing_data)

# Separate features and target label from training data
train_data = train_all_data.drop("price", axis=1)
train_data_label = train_all_data['price'].copy()  # Copy to avoid SettingWithCopyWarning

test_data = test_all_data.drop("price", axis=1)
test_data_label = test_all_data['price'].copy()  # Copy to avoid SettingWithCopyWarning

# # Uncomment to check shape and value counts
# print(housing_data.shape)
# print(housing_data['stories'].value_counts())
# print(housing_data['parking'].value_counts())

# # Example correlation matrix after encoding categorical variables
# train_data_encoded = pd.get_dummies(train_data, drop_first=True)
# corr_matrix = train_data_encoded.corr()
# print(corr_matrix['price'].sort_values(ascending=False))

# # Scatter matrix for visual analysis
# attribute = ["area", "bedrooms", "price"]
# s_matrix = scatter_matrix(train_data[attribute], figsize=(12, 8))
# print(s_matrix)

# Function to impute missing numeric values using median
def get_imputer_value(train_data):
    imputer = SimpleImputer(strategy="median")

    # Copy data to avoid modifying original dataframe
    imputed_data = train_data.copy()

    # Impute only numeric columns
    imputed_data= imputer.fit_transform(train_data)
    return imputed_data

# Apply imputation to training data (numeric columns)
imputer_output = get_imputer_value(train_data.select_dtypes(include=[np.number]))
# print(imputer_output.info())  # Check imputed dataframe info

# Function to create preprocessing pipeline for numeric and categorical features
def make_pipeline(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(exclude=[np.number]).columns

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


# Create preprocessing pipeline
pip = make_pipeline(train_data)

# Create regression model instead of classifier

model = RandomForestRegressor()
# model = DecisionTreeRegressor()
# model = LinearRegression()
# model = LGBMRegressor(verbose=1)
# Combine preprocessing and model into a single pipeline for training and cross-validation
full_pipeline = Pipeline(steps=[('preprocessor', pip),
                                ('regressor', model)])

# Train model on training data and labels
full_pipeline.fit(train_data, train_data_label)

# Select first 5 samples for prediction testing
some_data = test_data.iloc[:5]
some_label = test_data_label.iloc[:5]

# Predict house prices for the sample data
result = full_pipeline.predict(some_data)
# print(result)
# print(list(some_label))

def evaluate_model():
<<<<<<< HEAD
    housing_prediction = model.predict(train_data)
=======
    housing_prediction = full_pipeline.predict(train_data)
>>>>>>> 5984167ecfa68f4943eac877b5eb2e503b19fd5b
    mse = mean_squared_error(test_data_label, housing_prediction)
    rmse = np.sqrt(mse)
    return rmse

def score():
    # Use cross_val_score on full pipeline with proper scor ing parameter
    scores = cross_val_score(full_pipeline, test_data, test_data_label,
                             scoring="neg_mean_squared_error", cv=3)

    # Convert negative MSE to positive and sort
    score_result = np.sort(-scores)
    return score_result

# Removed all print statements related to price, RMSE, scores, and description

cv_scores = score()
print("\nCross-validation Scores:", cv_scores)
print("Mean CV MSE:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())
print("Train RMSE:", evaluate_model())
print("Cross-Validated RMSE:", np.sqrt(cv_scores.mean()))

