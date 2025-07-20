import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    df = pd.read_csv("CarPrice.csv")
    return df

def preprocess_data(df):
    features = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginesize', 'horsepower', 'citympg', 'highwaympg']
    df = df[features + ['price']].dropna()
    
    label_cols = ['fueltype', 'aspiration', 'carbody', 'drivewheel']
    encoders = {}
    
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y, encoders

@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def main():
    st.title("🚗 Car Price Predictor App")
    st.markdown("Estimate car price based on specifications using a trained machine learning model.")

    df = load_data()
    X, y, encoders = preprocess_data(df)
    model = train_model(X, y)

    st.sidebar.header("Enter Car Features")

    fueltype = st.sidebar.selectbox("Fuel Type", encoders['fueltype'].classes_)
    aspiration = st.sidebar.selectbox("Aspiration", encoders['aspiration'].classes_)
    carbody = st.sidebar.selectbox("Car Body", encoders['carbody'].classes_)
    drivewheel = st.sidebar.selectbox("Drive Wheel", encoders['drivewheel'].classes_)
    enginesize = st.sidebar.slider("Engine Size", 50, 400, 150)
    horsepower = st.sidebar.slider("Horsepower", 50, 300, 100)
    citympg = st.sidebar.slider("City MPG", 5, 60, 25)
    highwaympg = st.sidebar.slider("Highway MPG", 5, 60, 30)

    input_data = pd.DataFrame([{
        'fueltype': encoders['fueltype'].transform([fueltype])[0],
        'aspiration': encoders['aspiration'].transform([aspiration])[0],
        'carbody': encoders['carbody'].transform([carbody])[0],
        'drivewheel': encoders['drivewheel'].transform([drivewheel])[0],
        'enginesize': enginesize,
        'horsepower': horsepower,
        'citympg': citympg,
        'highwaympg': highwaympg
    }])

    if st.button("Predict Price"):
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Car Price: ${int(prediction):,}")

        st.subheader("🔍 Feature Importance")
        importance = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        importance.sort_values().plot(kind='barh', color='teal', ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
