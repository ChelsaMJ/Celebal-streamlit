import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Generate synthetic data
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Train linear regression model
def train_model():
    df = generate_house_data()
    X = df[['size_sqft']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Streamlit UI
def main():
    st.title('🏠 House Price Predictor')
    st.write("Enter the size of the house to estimate its price:")

    model = train_model()

    size = st.number_input('House Size (sqft)', min_value=500, max_value=5000, value=1500)

    if st.button('Predict Price'):
        prediction = model.predict([[size]])
        st.success(f'Estimated Price: ${prediction[0]:,.2f}')

        df = generate_house_data()
        fig = px.scatter(df, x='size_sqft', y='price', title='House Size vs Price')
        fig.add_scatter(x=[size], y=[prediction[0]], mode='markers',
                        marker=dict(color='red', size=12), name='Prediction')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
