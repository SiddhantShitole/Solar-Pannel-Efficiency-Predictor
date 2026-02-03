import streamlit as st
import pickle
import requests
import pandas as pd

# 1. Load Model
try:
    model = pickle.load(open('solar_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model not found. Run train_solar.py first.")
    st.stop()

st.set_page_config(page_title="Solar AI Manager", page_icon="☀️", layout="wide")
st.title("☀️ Smart Solar Forecaster")

# Sidebar
menu = st.sidebar.radio("Mode", ["🌍 Smart Forecast (Live API)", "🔮 Manual Simulation"])

# --- MODE 1: LIVE API (The New Feature) ---
if menu == "🌍 Smart Forecast (Live API)":
    st.header("Real-Time Solar Prediction 🌍")
    st.info("Enter a location to fetch live Weather & Irradiance data.")

    # 1. Get Location
    city = st.text_input("Enter City Name", "Pune")

    if st.button("Get Forecast"):
        try:
            # Step A: Get Coordinates (Lat/Lon)
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            geo_data = requests.get(geo_url).json()

            if not geo_data.get('results'):
                st.error("City not found.")
            else:
                lat = geo_data['results'][0]['latitude']
                lon = geo_data['results'][0]['longitude']
                name = geo_data['results'][0]['name']

                # Step B: Get Real-Time Solar & Weather Data
                meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,shortwave_radiation,wind_speed_10m"
                weather_data = requests.get(meteo_url).json()
                
                current = weather_data['current']
                
                # EXTRACT INPUTS
                air_temp = current['temperature_2m']
                irradiance_watts = current['shortwave_radiation'] # W/m²
                
                # Scale Irradiance for Model (Kaggle data was in kW/m² or similar scale)
                # If your slider was 0.0-1.2, we must divide Watts by 1000
                irr_input = irradiance_watts / 1000.0 

                # CALCULATE MODULE TEMP (Formula: Air + Sun_Heating)
                mod_temp = air_temp + (irradiance_watts * 0.025)

                # Step C: Display Data
                st.success(f"📍 Location: {name}")
                c1, c2, c3 = st.columns(3)
                c1.metric("☀️ Irradiance", f"{irradiance_watts} W/m²")
                c2.metric("🌡️ Air Temp", f"{air_temp} °C")
                c3.metric("🔥 Est. Panel Temp", f"{mod_temp:.1f} °C")

                # Step D: Predict
                prediction = model.predict([[irr_input, air_temp, mod_temp]])[0]

                st.divider()
                st.subheader(f"⚡ AI Predicted Output: {prediction:,.2f} kW")
                
                if prediction < 100 and irradiance_watts > 500:
                     st.warning("⚠️ Prediction is low. (Check model scaling)")

        except Exception as e:
            st.error(f"Error: {e}")

# --- MODE 2: MANUAL (Old Version) ---
elif menu == "🔮 Manual Simulation":
    st.header("🔮 Manual Simulation")
    
    col1, col2, col3 = st.columns(3)
    irr = col1.slider("Solar Irradiation (kW/m²)", 0.0, 1.2, 0.8)
    amb_temp = col2.slider("Air Temperature (°C)", 15.0, 45.0, 32.0)
    mod_temp = col3.slider("Module Temperature (°C)", 20.0, 75.0, 50.0)

    if st.button("Simulate"):
        pred = model.predict([[irr, amb_temp, mod_temp]])[0]
        st.success(f"Expected Power: {pred:,.2f} kW")