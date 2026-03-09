import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os

st.title("Iris Flower Prediction with Accuracy & Confusion Matrix")

# Load dataset safely
file_path = os.path.join(os.path.dirname(__file__), "Iris.csv")
data = pd.read_csv(file_path)

X = data.drop(["Id","Species"], axis=1)
y = data["Species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Predict on test data
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display metrics
st.subheader("Model Accuracy")
st.write(f"{accuracy*100:.2f}%")

st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(cm)
st.dataframe(cm_df)

# User input for prediction
st.subheader("Enter Flower Measurements")
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    st.success(f"Predicted Flower: {prediction[0]}")
