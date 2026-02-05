import streamlit as st
import pickle
import requests
import pandas as pd

#Load Model
try:
    model = pickle.load(open('solar_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model not found. Run train_solar.py first.")
    st.stop()

st.set_page_config(page_title="Solar AI Manager", page_icon="☀️", layout="wide")
st.title("☀️ Smart Solar Forecaster & Fault Detector")

# Sidebar
menu = st.sidebar.radio("Mode", ["🌍 Smart Forecast (Live API)", "🔮 Manual Simulation"])

# MODE 1: LIVE API (Upgraded)
if menu == "🌍 Smart Forecast (Live API)":
    st.header("Real-Time Prediction & Efficiency Check 🌍")
    st.info("Step 1: Get the AI Prediction based on live weather.")

    # 1. Get Location
    city = st.text_input("Enter City Name", "Pune")

    if st.button("Get Forecast"):
        try:
            # Step A: Get Coordinates
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            geo_data = requests.get(geo_url).json()

            if not geo_data.get('results'):
                st.error("City not found.")
            else:
                lat = geo_data['results'][0]['latitude']
                lon = geo_data['results'][0]['longitude']
                name = geo_data['results'][0]['name']

                # Step B: Get Weather
                meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,shortwave_radiation,wind_speed_10m"
                weather_data = requests.get(meteo_url).json()
                current = weather_data['current']
                
                # EXTRACT INPUTS
                air_temp = current['temperature_2m']
                irradiance_watts = current['shortwave_radiation']
                irr_input = irradiance_watts / 1000.0  # Convert to kW/m² for model
                mod_temp = air_temp + (irradiance_watts * 0.025)

                # Step C: Show Weather
                st.success(f"📍 Location: {name}")
                c1, c2, c3 = st.columns(3)
                c1.metric("☀️ Irradiance", f"{irradiance_watts} W/m²")
                c2.metric("🌡️ Air Temp", f"{air_temp} °C")
                c3.metric("🔥 Est. Panel Temp", f"{mod_temp:.1f} °C")

               # Step D: Predict
                raw_prediction = model.predict([[irr_input, air_temp, mod_temp]])[0]
                
                # SCALE DOWN: Convert "Power Plant" size to "Home" size (5kW system)
                prediction = raw_prediction / 45000 
                
                # Store prediction in session state
                st.session_state['prediction'] = prediction
                st.session_state['has_predicted'] = True

        except Exception as e:
            st.error(f"Error: {e}")

    # FEATURE: EFFICIENCY CHECKER 
    # We check if a prediction exists in the session
    if st.session_state.get('has_predicted'):
        pred = st.session_state['prediction']
        
        st.divider()
        st.subheader(f"⚡ AI Expected Output: {pred:,.2f} kW")
        
        st.info("Step 2: Enter your actual inverter reading to check efficiency.")
        
        # User enters actual value (Small numbers now, like 3.5 or 4.2)
        actual = st.number_input("🔌 Actual Power Generation (kW)", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
        if actual > 0:
            # Logic: Calculate Efficiency
            efficiency = (actual / pred) * 100
            loss = pred - actual
            
            st.write(f"### Efficiency Rating: {efficiency:.1f}%")

            # The "Good or Bad" Logic
            if efficiency >= 90:
                st.success("✅ EXCELLENT: System is performing optimally!")
            elif efficiency >= 80:
                st.warning(f"⚠️ FAIR: You are losing {loss:.2f} kW. Consider cleaning panels.")
            else:
                st.error(f"🚨 CRITICAL: High Efficiency Loss! ({loss:.2f} kW missing). Check for faults.")
            
            # Progress Bar for Visual Appeal
            st.progress(min(efficiency / 100, 1.0))

# MODE 2: MANUAL 
elif menu == "🔮 Manual Simulation":
    st.header("🔮 Manual Simulation")
    col1, col2, col3 = st.columns(3)
    irr = col1.slider("Solar Irradiation (kW/m²)", 0.0, 1.2, 0.8)
    amb_temp = col2.slider("Air Temperature (°C)", 15.0, 45.0, 32.0)
    mod_temp = col3.slider("Module Temperature (°C)", 20.0, 75.0, 50.0)

    if st.button("Simulate"):
        raw_pred = model.predict([[irr, amb_temp, mod_temp]])[0]
        
        # SAME SCALING here so both modes match!
        pred = raw_pred / 45000  
        
        st.success(f"Expected Power: {pred:,.2f} kW")