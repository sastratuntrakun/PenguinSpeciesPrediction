import streamlit as st
import pandas as pd
import pickle

# Load the saved model and encoders
with open('model_penguin_66130701725.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Create the Streamlit app
st.title('Penguin Species Prediction App')

# Input fields for user to enter penguin data
island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0)
culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0)
flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0)
body_mass_g = st.number_input('Body Mass (g)', min_value=0.0)
sex = st.selectbox('Sex', ['MALE', 'FEMALE'])

# Create a button to trigger prediction
if st.button('Predict Species'):
    # Create a DataFrame with the user's input
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })

    # Encode categorical features
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])

    # Make a prediction using the loaded model
    y_pred_new = model.predict(x_new)

    # Decode the prediction back to the original species name
    result = species_encoder.inverse_transform(y_pred_new)

    # Display the prediction
    st.write('Predicted Penguin Species:', result[0])
