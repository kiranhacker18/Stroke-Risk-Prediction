import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from data_preprocessing import load_data, preprocess_data

# Load and preprocess data
df = load_data()
df = preprocess_data(df)

# Split data
X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier()
lr_model = LogisticRegression(max_iter=1000)
knn_model = KNeighborsClassifier()

# Train models
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Accuracy
rf_acc = rf_model.score(X_test, y_test)
lr_acc = lr_model.score(X_test, y_test)
knn_acc = knn_model.score(X_test, y_test)

print(f"Random Forest Accuracy: {rf_acc}")
print(f"Logistic Regression Accuracy: {lr_acc}")
print(f"KNN Accuracy: {knn_acc}")

# 🔥 Save models
base_path = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(base_path, exist_ok=True)

with open(os.path.join(base_path, 'rf_model.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)

with open(os.path.join(base_path, 'lr_model.pkl'), 'wb') as f:
    pickle.dump(lr_model, f)

with open(os.path.join(base_path, 'knn_model.pkl'), 'wb') as f:
    pickle.dump(knn_model, f)

print("✅ All models saved successfully!")

