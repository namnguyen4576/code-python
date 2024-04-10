# Importing necessary libraries
import pandas as pd
import numpy as np

# Load the raw data into a DataFrame
raw_data = pd.read_csv('telecom_data.csv')

# Display the first few rows of the DataFrame
print(raw_data.head())

# Check for missing values
print(raw_data.isnull().sum())

# Impute missing values with appropriate methods
raw_data['customer_age'].fillna(raw_data['customer_age'].median(), inplace=True)
raw_data['monthly_charges'].fillna(raw_data['monthly_charges'].mean(), inplace=True)

# Remove outliers
Q1 = raw_data['monthly_charges'].quantile(0.25)
Q3 = raw_data['monthly_charges'].quantile(0.75)
IQR = Q3 - Q1
raw_data = raw_data[(raw_data['monthly_charges'] > (Q1 - 1.5 * IQR)) & (raw_data['monthly_charges'] < (Q3 + 1.5 * IQR))]

# Encode categorical variables
encoded_data = pd.get_dummies(raw_data, columns=['contract_type', 'gender'], drop_first=True)

# Standardize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data[['customer_age', 'monthly_charges']])

# Replace original features with scaled features
encoded_data[['customer_age', 'monthly_charges']] = scaled_data

# Save cleaned and preprocessed data to a new CSV file
encoded_data.to_csv('cleaned_telecom_data.csv', index=False)
