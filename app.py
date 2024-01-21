import streamlit as st
from PIL import Image
import numpy as np
import xgboost as xgb

# Function to preprocess and classify the image
def predict_house_price(features, model):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(xgb.DMatrix(input_data))
    return prediction[0]

def main():
    st.title("Machine Learning Based Housing Price Predictor")
    st.header("XGBoost Housing Price Predictor")
    st.text("Enter the parameters to predict your house's price:")

    # Load the model
    model_file_path = 'xgboostBestModel.model'
    loaded_booster = xgb.Booster()
    loaded_booster.load_model(model_file_path)

    # Input options for house features
    int_sqft = st.slider("Enter INT_SQFT", min_value=500, max_value=5000, value=1500, step=100)
    n_rooms = st.slider("Enter N_ROOMS", min_value=1, max_value=10, value=3)
    n_bathrooms = st.slider("Enter N_BATHROOMS", min_value=1, max_value=5, value=2)
    n_sale_cond = st.selectbox("Select N_SALE_COND", ["Normal", "Abnormal", "AdjLand", "Partial"])

    # Convert categorical feature to numerical
    n_sale_cond_mapping = {"Normal": 0, "Abnormal": 1, "AdjLand": 2, "Partial": 3}
    n_sale_cond_numeric = n_sale_cond_mapping[n_sale_cond]

    # Make prediction
    features = [int_sqft, n_rooms, n_bathrooms, n_sale_cond_numeric]
    predicted_price = predict_house_price(features, loaded_booster)

    # Display the result
    st.subheader("House Features:")
    st.write(f"INT_SQFT: {int_sqft}")
    st.write(f"N_ROOMS: {n_rooms}")
    st.write(f"N_BATHROOMS: {n_bathrooms}")
    st.write(f"N_SALE_COND: {n_sale_cond}")

    st.subheader("Predicted House Price:")
    st.write(f"${predicted_price:,.2f}")

if __name__ == "__main__":
    main()
