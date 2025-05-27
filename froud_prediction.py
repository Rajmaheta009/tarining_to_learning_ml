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
from xgboost import XGBClassifier

# Load data
print("🔄 Loading dataset...")
fraud_data = pd.read_csv("csv_database/Fraud.csv")
print("✅ Dataset loaded. Shape:", fraud_data.shape)

# Split data
def train_test_data(data):
    print("🔄 Splitting data into train and test sets...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    print("✅ Train set:", train_data.shape, "| Test set:", test_data.shape)
    return train_data, test_data

train_data, test_data = train_test_data(fraud_data)

# Separate target labels
print("🔄 Separating target labels...")
train_data_labels = train_data['isFlaggedFraud'].copy()
test_data_labels = test_data['isFlaggedFraud'].copy()

x_train = train_data.drop("isFlaggedFraud", axis=1)
x_test = test_data.drop("isFlaggedFraud", axis=1)

# Detect column types
print("🔍 Detecting numerical and categorical columns...")
num_cols = x_train.select_dtypes(include=['int64','float64']).columns
cat_cols = x_train.select_dtypes(include=['object']).columns
print("🔢 Numerical columns:", list(num_cols))
print("🔠 Categorical columns:", list(cat_cols))

# Preprocessing pipelines
print("🔧 Setting up preprocessing pipelines...")
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Fit and transform
print("🔄 Applying preprocessing pipelines...")
x_train_imputed = full_pipeline.fit_transform(x_train)
x_test_imputed = full_pipeline.transform(x_test)
print("✅ Preprocessing complete. Transformed shape:", x_train_imputed.shape)

# Select model
# model = DecisionTreeClassifier()
# model = LogisticRegression()
print("🧠 Initializing Random Forest classifier...")
model = RandomForestClassifier(class_weight='balanced', n_jobs=10, n_estimators=100,criterion='log_loss')
# print("🧠 Initializing XGBoost classifier...")
# model = XGBClassifier(tree_method='hist', device='cpu')


# Train model
print("🏋️ Training model...")
model.fit(x_train_imputed, train_data_labels)
print("✅ Model training complete.")

# Evaluation functions
def evaluate_model(model):
    print("📊 Evaluating model on training set...")
    fraud_prediction = model.predict(x_train_imputed)
    mse = mean_squared_error(train_data_labels, fraud_prediction)
    rmse = np.sqrt(mse)
    print(f"📈 RMSE on training set: {rmse:.4f}")
    return rmse

def score(model):
    print("🔁 Performing 3-fold cross-validation...")
    scores = cross_val_score(model, x_test_imputed, test_data_labels,
                             scoring="neg_mean_squared_error", cv=3)
    score_result = np.sort(-scores)
    print("📊 Cross-validation MSE scores:", score_result)
    return score_result

# Run evaluation
s_var = score(model)
e_var = evaluate_model(model)

# Final output
print("\n🔍 Final Evaluation Metrics:")
print("✔️ RMSE (Train):", e_var)
print("✔️ Cross-Validation MSE:", s_var)

# Save model
print("💾 Saving model to 'pkl files/fraud.pkl'...")
joblib.dump((model, full_pipeline), "pkl files/fraud.pkl")
print("✅ Model saved successfully.")
