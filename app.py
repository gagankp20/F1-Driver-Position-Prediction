import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="F1 Oracle: Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

# --- Load Model and Data ---
@st.cache_data
def load_data():
    with open('f1_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    features_df = pd.read_csv('f1_features.csv')
    return model, features_df

ml_model, features = load_data()

# --- Google AI API Configuration ---
API_KEY = "AIzaSyD0vOsX4Q1Do8UhDrtxRg7cKc8GNjvOslA" # ❗️ PASTE YOUR KEY HERE
genai.configure(api_key=API_KEY)

# --- App Header ---
st.title("🏎️ F1 Oracle: Race Performance Predictor")
st.markdown("Select a driver, team, and starting position to get a race prediction.")

# --- User Input (with Dependent Dropdowns) ---
col1, col2, col3 = st.columns(3)

with col1:
    driver_name = st.selectbox("Select a Driver:", sorted(features['driver_name'].unique()))

# --- NEW: Team dropdown is now dependent on the selected driver ---
with col2:
    # Filter the dataframe to get only the teams the selected driver has driven for
    available_teams = sorted(features[features['driver_name'] == driver_name]['constructor_name'].unique())
    team_name = st.selectbox("Select a Team:", available_teams)

with col3:
    start_pos = st.slider("Select a Starting Position:", 1, 20, 10)

# --- Prediction Logic ---
if st.button("Get Prediction"):
    # Find the latest stats for the selected driver to use as a base
    latest_stats = features[features['driver_name'] == driver_name].iloc[-1:].copy()

    if not latest_stats.empty:
        # Update the qualifying position with the user's input
        latest_stats['qualifying_position'] = start_pos
        
        # Prepare data for prediction (ensure columns match model's training)
        model_features = [
            'year', 'round', 'circuitid', 'qualifying_position', 'driver_age',
            'driver_points_rolling_5', 'driver_position_rolling_5',
            'constructor_points_rolling_5'
        ]
        prediction_features = latest_stats[model_features]
        
        predicted_position = ml_model.predict(prediction_features)[0]

        st.subheader(f"Prediction for {driver_name} ({team_name})")
        st.metric(label="Predicted Finishing Position", value=f"P{int(round(predicted_position))}")

        # --- AI Commentary ---
        with st.spinner("🤖 Generating AI Analyst Commentary..."):
            prompt_template = f"""
            You are an expert F1 sports analyst. Based on the following data, write a 2-3 sentence race prediction summary.

            - Driver: {driver_name}
            - Team: {team_name}
            - Starting Position: {start_pos}
            - Recent Average Finish: {latest_stats['driver_position_rolling_5'].iloc[0]:.1f}
            
            Our machine learning model predicts a finishing position of: {int(round(predicted_position))}
            """
            
            genai_model = genai.GenerativeModel('gemini-pro-latest')
            response = genai_model.generate_content(prompt_template)
            
            st.success("AI Commentary:")
            st.write(response.text)
    else:
        st.error("No historical data available for this driver to make a prediction.")