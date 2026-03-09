import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))

st.title("Iris Flower Prediction")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):

    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

    st.success(f"Predicted Flower: {prediction[0]}")
