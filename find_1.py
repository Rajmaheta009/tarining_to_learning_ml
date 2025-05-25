import pandas as pd

data = pd.read_csv("csv_database/Housing.csv")

# Filter rows where isFlaggedFraud == 1
fraud_rows = data[data['price'] >= 7000000]

print(fraud_rows)
