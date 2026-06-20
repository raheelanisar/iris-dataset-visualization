import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Prediction App")

# Load Dataset
df = pd.read_csv("Iris.csv")

st.subheader("Dataset")
st.write(df.head())

# Train Model Button
if st.button("Train Model"):

    X = df.iloc[:, 1:5]
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    pickle.dump(model, open("iris_model.pkl", "wb"))

    st.success("Model Trained and Saved Successfully!")

# Load Model
try:
    model = pickle.load(open("iris_model.pkl", "rb"))

    st.subheader("Enter Flower Measurements")

    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")

    if st.button("Predict"):

        prediction = model.predict([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        st.success(f"Predicted Species: {prediction[0]}")

except:
    st.warning("Please Train Model First")

# Visualization
st.subheader("Dataset Visualization")

fig, ax = plt.subplots()

ax.scatter(
    df["SepalLengthCm"],
    df["PetalLengthCm"]
)

ax.set_xlabel("Sepal Length")
ax.set_ylabel("Petal Length")

st.pyplot(fig)