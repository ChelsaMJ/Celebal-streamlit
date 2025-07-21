# Car Price Predictor App
#### Week 7 - Deploying Machine Learning Models with Streamlit

**Link to view the working Streamlit app: https://celebal-app-ka8vn2motzp8rnmkxkngaj.streamlit.app/**

This is a Streamlit web application that predicts the price of a car based on its specifications using a trained Random Forest Regressor model. The app allows users to input car features like fuel type, horsepower, mileage, and more, and returns an estimated car price.

---

## Features

- Interactive sidebar for feature input
- Machine Learning model: `RandomForestRegressor`
- Data preprocessing using `LabelEncoder`
- Visualization of feature importance using `matplotlib`
- Fully interactive and deployed using Streamlit

---

## ML Model Details

- **Model**: Random Forest Regressor (`sklearn`)
- **Input Features**:
  - `fueltype`
  - `aspiration`
  - `carbody`
  - `drivewheel`
  - `enginesize`
  - `horsepower`
  - `citympg`
  - `highwaympg`
- **Target Variable**: `price`

---

## How to Run the App Locally

### 1. Clone the repo
```
git clone https://github.com/your-username/car-price-predictor.git
cd car-price-predictor
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```
streamlit run app.py
```
