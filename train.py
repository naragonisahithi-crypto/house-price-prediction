import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the data
df = pd.read_csv("data/house_data.csv")

# Convert location to numbers
df['location'] = df['location'].map({'Urban': 2, 'Semi-Urban': 1, 'Rural': 0})

# Features & target
X = df[['area', 'bedrooms', 'bathrooms', 'location']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved successfully!")
