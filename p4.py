import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1) Generate synthetic interactions
np.random.seed(42)
n_customers     = 50
n_products      = 15
n_interactions  = 500

customers  = [f"CUST_{i:03d}" for i in range(n_customers)]
products   = [f"PROD_{i:03d}" for i in range(n_products)]
categories = ['Electronics', 'Clothing', 'Home', 'Sports']
brands     = ['BrandA', 'BrandB', 'BrandC', 'BrandD']

data = {
    'Customer_ID': np.random.choice(customers, size=n_interactions),
    'Product':     np.random.choice(products,  size=n_interactions),
    'Date_of_Purchase': pd.to_datetime('2024-01-01') +
                        pd.to_timedelta(np.random.randint(0, 365, size=n_interactions), unit='d'),
    'Frequency':       np.random.randint(1, 6, size=n_interactions),
    'Annual_Income':   np.random.choice([30000, 50000, 70000, 90000], size=n_interactions),
    'Gender':          np.random.choice(['Male','Female'], size=n_interactions),
    'Age_Group':       np.random.choice(['18-25','26-35','36-45','46-55'], size=n_interactions),
    'Region':          np.random.choice(['North','South','East','West'], size=n_interactions),
    'Price':           np.random.uniform(10, 1000, size=n_interactions).round(2),
    'Category':        np.random.choice(categories, size=n_interactions),
    'Brand':           np.random.choice(brands,    size=n_interactions),
}

df_synth = pd.DataFrame(data)

# 2) Derive recency feature
today = pd.to_datetime('2025-05-24')
df_synth['Days_Since_Last_Purchase'] = (today - df_synth['Date_of_Purchase']).dt.days

# 3) Label-encode categorical columns
for col in ['Gender','Age_Group','Region','Category','Brand']:
    df_synth[f"{col}_Label"] = LabelEncoder().fit_transform(df_synth[col])

# 4) Print a sample and save to CSV
print("=== Synthetic Expanded Purchase Data Sample ===")
print(df_synth.head(10).to_string(index=False))

# Optional: save for later use in your modeling script
df_synth.to_csv('synthetic_purchase_data.csv', index=False)
print("\nSaved synthetic data to synthetic_purchase_data.csv")
