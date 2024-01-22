import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd

# Function to preprocess and classify the house price
def predict_house_price(features, model, category_mapping):
    # Encode categorical features
    input_data = encode_categorical_features(features, category_mapping)
    input_data = np.array(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(xgb.DMatrix(input_data))
    return prediction[0]

# Function to encode categorical features
def encode_categorical_features(input_features, category_mapping):
    encoded_data = pd.DataFrame()

    for column, mapping in category_mapping.items():
        user_input = input_features.get(column)
        if user_input is not None:
            for category, value in mapping.items():
                encoded_data[f"{column}_{category}"] = 1 if user_input == category else 0

    # Include numerical features
    numerical_features = ['INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'AGE']
    for num_feature in numerical_features:
        if num_feature in input_features:
            encoded_data[num_feature] = input_features[num_feature]

    return encoded_data

def main():
    st.title("Machine Learning Based Housing Price Predictor")
    st.header("XGBoost Housing Price Predictor")
    st.text("Enter the parameters to predict your house's price:")

    # Load the model
    model_file_path = 'xgboostBestModel.h5'
    loaded_booster = xgb.Booster()
    loaded_booster.load_model(model_file_path)

    # Define category mapping for encoding
    category_mapping = {
        'AREA': {'Adyar': 0, 'Anna Nagar': 1, 'Chrompet': 2, 'KK Nagar': 3, 'Karapakam': 4, 'T Nagar': 5, 'Velachery': 6},
        'SALE_COND': {'AbNormal': 0, 'Adj Land': 1, 'Family': 2, 'Normal Sale': 3, 'Partial': 4},
        'PARK_FACIL': {'No': 0, 'Yes': 1},
        'BUILDTYPE': {'Commercial': 0, 'House': 1, 'Others': 2},
        'UTILITY_AVAIL': {'All Pub': 0, 'ELO': 1, 'NoSeWa': 2, 'NoSewr ': 3},
        'STREET': {'Gravel': 0, 'No Access': 1, 'Paved': 2}
    }

    # Input options for house features
    int_sqft = st.slider("Enter INT_SQFT", min_value=500, max_value=5000, value=1500, step=10)
    dist_mainroad = st.slider("Enter DIST_MAINROAD", min_value=1, max_value=100, value=50)
    n_bedroom = st.slider("Enter N_BEDROOM", min_value=1, max_value=10, value=3)
    n_bathroom = st.slider("Enter N_BATHROOM", min_value=1, max_value=5, value=2)
    n_room = st.slider("Enter N_ROOM", min_value=1, max_value=10, value=3)
    age = st.slider("Enter AGE", min_value=1, max_value=100, value=20)

    # Define the features that come from select boxes
    categorical_features = [
        'AREA', 'SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET'
    ]

    # Prepare input features
    input_features = {
        'INT_SQFT': int_sqft,
        'DIST_MAINROAD': dist_mainroad,
        'N_BEDROOM': n_bedroom,
        'N_BATHROOM': n_bathroom,
        'N_ROOM': n_room,
        'AGE': age
    }

    # Dynamically add one-hot encoded features based on user selection
    for feature in categorical_features:
        selected_value = st.selectbox(f"Select {feature}", category_mapping[feature].keys())
        input_features.update({feature: selected_value})

    # Make prediction
    predicted_price = predict_house_price(input_features, loaded_booster, category_mapping)

    # Display the result
    st.subheader("House Features:")
    for feature, value in input_features.items():
        st.write(f"{feature}: {value}")

    st.subheader("Predicted House Price:")
    st.write(f"${predicted_price:,.2f}")

if __name__ == "__main__":
    main()
# import streamlit as st
# from PIL import Image
# import numpy as np
# import xgboost as xgb

# # Function to preprocess and classify the image
# def predict_house_price(features, model):
#     input_data = np.array(features).reshape(1, -1)
#     prediction = model.predict(xgb.DMatrix(input_data))
#     return prediction[0]

# def main():
#     st.title("Machine Learning Based Housing Price Predictor")
#     st.header("XGBoost Housing Price Predictor")
#     st.text("Enter the parameters to predict your house's price:")

#     # Load the model
#     model_file_path = 'xgboostBestModel.model'
#     loaded_booster = xgb.Booster()
#     loaded_booster.load_model(model_file_path)

#     # Input options for house features
#     int_sqft = st.slider("Enter INT_SQFT", min_value=500, max_value=5000, value=1500, step=10)
#     n_rooms = st.slider("Enter N_ROOMS", min_value=1, max_value=10, value=3)
#     n_bathrooms = st.slider("Enter N_BATHROOMS", min_value=1, max_value=5, value=2)
#     n_sale_cond = st.selectbox("Select N_SALE_COND", ["Normal", "Abnormal", "AdjLand", "Partial"])

#     # Convert categorical feature to numerical
#     n_sale_cond_mapping = {"Normal": 0, "Abnormal": 1, "AdjLand": 2, "Partial": 3}
#     n_sale_cond_numeric = n_sale_cond_mapping[n_sale_cond]

#     # Make prediction
#     features = [int_sqft, n_rooms, n_bathrooms, n_sale_cond_numeric]
#     predicted_price = predict_house_price(features, loaded_booster)

#     # Display the result
#     st.subheader("House Features:")
#     st.write(f"INT_SQFT: {int_sqft}")
#     st.write(f"N_ROOMS: {n_rooms}")
#     st.write(f"N_BATHROOMS: {n_bathrooms}")
#     st.write(f"N_SALE_COND: {n_sale_cond}")

#     st.subheader("Predicted House Price:")
#     st.write(f"${predicted_price:,.2f}")

# if __name__ == "__main__":
#     main()
