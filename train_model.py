import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
df = pd.read_csv('house_prices.csv')

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Preprocessing for categorical features
categorical_features = ['lawn', 'location_type']
numeric_features = ['bedrooms', 'bathrooms', 'kitchens', 'area_sqft']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', 'passthrough', numeric_features)
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'house_price_model.pkl')
print('Model trained and saved as house_price_model.pkl') 