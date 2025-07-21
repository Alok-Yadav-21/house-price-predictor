import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
data_size = 1000

# Generate features
data = {
    'bedrooms': np.random.randint(1, 6, data_size),
    'bathrooms': np.random.randint(1, 4, data_size),
    'kitchens': np.random.randint(1, 3, data_size),
    'lawn': np.random.choice(['Yes', 'No'], data_size),
    'area_sqft': np.random.randint(500, 5000, data_size),
    'location_type': np.random.choice(['City', 'Out of City'], data_size)
}

df = pd.DataFrame(data)

# Generate target variable (price) with some logic
base_price = 50000

def calculate_price(row):
    price = base_price
    price += row['bedrooms'] * 20000
    price += row['bathrooms'] * 15000
    price += row['kitchens'] * 10000
    price += row['area_sqft'] * 50
    price += 30000 if row['lawn'] == 'Yes' else 0
    price += 50000 if row['location_type'] == 'City' else -20000
    # Add some noise
    price += np.random.normal(0, 20000)
    return price

df['price'] = df.apply(calculate_price, axis=1)

df.to_csv('house_prices.csv', index=False)
print('Synthetic dataset saved as house_prices.csv') 