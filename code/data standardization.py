import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read the data from the CSV file
file_path = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/train 100.csv'
data = pd.read_csv(file_path)

# Identify continuous numerical columns (e.g., float64)
continuous_columns = data.select_dtypes(include=['float64']).columns

# Standardize only the continuous numerical columns
scaler = StandardScaler()
data[continuous_columns] = scaler.fit_transform(data[continuous_columns])

# Save the standardized data to a new file
standardized_file_path = 'C:/Users/Admin/Documents/0TL/ml/BTL/data/train 100 standardized.csv'
data.to_csv(standardized_file_path, index=False)

# Display the first few rows of the standardized data
data.head()
