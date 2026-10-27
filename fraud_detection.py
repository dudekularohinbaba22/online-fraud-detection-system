# Online Fraud Detection System
# Developed by Dudekula Rohin Baba

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

print("Loading dataset...")
# Creating sample data - replace with real creditcard.csv
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.99, 0.01], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Trained Successfully!")
print(f"Fraud Detection Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))

def predict_fraud(transaction_data):
    result = model.predict([transaction_data])
    return "FRAUDULENT" if result[0]==1 else "SAFE"
