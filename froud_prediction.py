import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

fraud_data = pd.read_csv("csv_database/Fraud.csv")

def train_test_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42,shuffle=True)
    return train_data, test_data

train_data, test_data = train_test_data(fraud_data)

train_data_labels = train_data['isFlaggedFraud'].copy()
test_data_labels = test_data['isFlaggedFraud'].copy()

x_train = train_data.drop("isFlaggedFraud", axis=1)
x_test = test_data.drop("isFlaggedFraud", axis=1)

num_cols = x_train.select_dtypes(include=['int64','float64']).columns
cat_cols = x_train.select_dtypes(include=['object']).columns

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


x_train_imputed = full_pipeline.fit_transform(x_train)
x_test_imputed = full_pipeline.transform(x_test)

# model = DecisionTreeClassifier()
# model = LogisticRegression()
model = RandomForestClassifier(class_weight='balanced')

model.fit(x_train_imputed, train_data_labels)

def evaluate_model(model):
    fraud_prediction = model.predict(x_train_imputed)
    mse = mean_squared_error(train_data_labels, fraud_prediction)
    rmse = np.sqrt(mse)
    return rmse

def score(model):
    scores = cross_val_score(model, x_test_imputed, test_data_labels,
                             scoring="neg_mean_squared_error", cv=3)

        # Convert negative MSE to positive and sort
    score_result = np.sort(-scores)
    return score_result

s_var = score(model)
e_var= evaluate_model(model)


print("Train Accuracy:",e_var)
print("Cross-validated Accuracy:", s_var)

joblib.dump((model,full_pipeline),"pkl files/fraud.pkl")