import pandas as pd
import random
from datetime import datetime, timedelta

# Load the dataset
data = pd.read_csv('EDA_Example.csv')
df = pd.DataFrame(data)

# Generate random dates between two dates (e.g., "2022-01-01" to "2025-12-31")
start_date = datetime(2022, 1, 1)
end_date = datetime(2025, 12, 31)

def generate_random_date(start_date, end_date):
    return start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

# Add a date column
df['date'] = [generate_random_date(start_date, end_date) for _ in range(len(df))]

# Display the updated dataset
print(df.head())

# Save the dataset to a CSV file   
df.to_csv('example_dataset.csv', index=False)