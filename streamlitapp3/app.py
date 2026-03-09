import streamlit as st
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))
st.title("Iris Flower Prediction")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):

    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

    st.success(f"Predicted Flower: {prediction[0]}")
    from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(accuracy)

st.subheader("Confusion Matrix")
st.write(cm)
