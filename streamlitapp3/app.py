import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title("Iris Flower Prediction")

# Load dataset
data = pd.read_csv("Iris.csv")

# Features and target
X = data.drop(["Id","Species"], axis=1)
y = data["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = pickle.load(open("model.pkl","rb"))

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Show accuracy
st.subheader("Model Accuracy")
st.write(f"{accuracy*100:.2f}%")

# Show confusion matrix
st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(cm)
st.dataframe(cm_df)

# User input for prediction
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Prediction button
if st.button("Predict"):
    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    st.success(f"Predicted Flower: {prediction[0]}")
