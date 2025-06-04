import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

car_data = pd.read_csv("csv_database/car__sales__data.csv")
# car_data = pd.read_csv("car__sales__data.csv")


# print(car_data['Condition'].values)


def train_test_dived(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

train_data, test_data = train_test_dived(car_data)

# print(train_data.head)
test_data_labels = test_data['Accident'].copy()
train_data_labels = train_data['Accident'].copy()

train_data_labels = train_data_labels.map({'Yes': 1, 'No': 0})
test_data_labels = test_data_labels.map({'Yes': 1, 'No': 0})


train_updated_data= train_data.drop("Accident",axis=1)
test_updated_data= test_data.drop("Accident",axis=1)


# print(train_data_labels.values_counts())
num_cols = train_updated_data.select_dtypes(include=['int64','float64']).columns
cat_cols = train_updated_data.select_dtypes(include=['object']).columns

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


train_data_imputed = full_pipeline.fit_transform(train_updated_data)
test_data_imputed = full_pipeline.transform(test_updated_data)

# model= LogisticRegression(max_iter=1000)
# model = DecisionTreeClassifier()
model = RandomForestClassifier(class_weight='balanced')
# model = XGBClassifier()

model.fit(train_data_imputed, train_data_labels)

def evaluate_model(model):
    car_accident_prediction = model.predict(train_data_imputed)
    accuracy = accuracy_score(train_data_labels, car_accident_prediction)
    return accuracy

def score(model):
    scores = cross_val_score(model, test_data_imputed, test_data_labels,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean()


s_var = score(model)
e_var= evaluate_model(model)

print("Train RMSE:", e_var)
print("Cross-validated RMSE:", s_var)


joblib.dump((model,full_pipeline), "pkl_files/car_accident_prediction.pkl")