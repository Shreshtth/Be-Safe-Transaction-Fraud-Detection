import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Function to generate a synthetic transaction
def generate_transaction(i, fraud_probability=0.30):
    is_fraud = np.random.rand() < fraud_probability
    status = 0 if is_fraud else 1
    transaction_type = np.random.choice(['UPI', 'RTGS', 'NEFT'])
    transaction_amount = np.random.uniform(1, 15000) if not is_fraud else np.random.uniform(1000, 20000)
    hour = np.random.randint(0, 24)
    specific_dates = pd.date_range(start='2024-03-01', end='2024-03-10')
    date = np.random.choice(specific_dates)  # Randomly select from the list of specific dates
    payer_vpa = np.random.randint(1, 101)  # Randomly select from 100 unique payer VPA
    payee_vpa = np.random.randint(1, 34)  # Randomly select from 33 unique payee VPA
    ip_address = f'{np.random.randint(0, 255)}.{np.random.randint(0, 255)}'

    return {
        'txnAmt': transaction_amount,
        'transaction_type': transaction_type,
        'status': status,
        'payer_vpa': payer_vpa,
        'payee_vpa': payee_vpa,
        'ip_address': ip_address,
        'hour': hour,
        'date': date
    }


# Generate dataset with 10000 transactions
num_transactions = 10000
transactions = [generate_transaction(i) for i in range(num_transactions)]

# Convert to DataFrame
df = pd.DataFrame(transactions)

# Convert txnAmt to float
df['txnAmt'] = df['txnAmt'].astype(int)
# The idea is to represent ip_address as a state and region information.
df['ip_address'] = df['ip_address'].astype(float)

label_encoder = LabelEncoder()

# Fit and transform the transaction_type column
# We are mapping: 0-UPI, 2-NEFT, 1-RTGS
df['transaction_type_encoded'] = label_encoder.fit_transform(df['transaction_type'])
# Drop the original transaction_type column
df.drop('transaction_type', axis=1, inplace=True)
# Note for improvement: Encode payer_vpa and payee_vpa using frequency encoding

# Feature Engineering
# Define time window for transaction frequency calculation (e.g., daily)
time_window = '1D'  # 1 day

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Group transactions by payer and payee VPA combination within the time window
grouped = df.groupby(['payer_vpa', 'payee_vpa', pd.Grouper(key='date', freq=time_window)])

# Calculate transaction frequency within each group
transaction_frequency = grouped.size().reset_index(name='transaction_frequency')

# Merge transaction frequency feature back into the DataFrame
df = pd.merge(df, transaction_frequency, on=['payer_vpa', 'payee_vpa', 'date'], how='left')

# Fill missing transaction frequencies with 1
df['transaction_frequency'] = df['transaction_frequency'].fillna(1)
# Convert to datetime format to integer
df['date'] = (df['date'] - pd.Timestamp("2024-01-01")).dt.days

# Future addition : If the transaction frequency is above threshold,
# don't allow more transactions in the same time window.

print(df.head())

df.to_csv('Data.csv', index=False)

# Load the dataset
df = pd.read_csv('Data.csv')
