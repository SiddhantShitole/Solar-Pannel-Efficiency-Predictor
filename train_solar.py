import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load Data
print("Loading datasets...")
try:
    gen_data = pd.read_csv('Plant_1_Generation_Data.csv')
    weather_data = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
except FileNotFoundError:
    print("❌ Error: Files not found! Make sure both CSVs are in this folder.")
    exit()

# 2. Format Dates (Crucial Step for merging)
# The dataset uses a specific format like '15-05-2020 12:00'
print("Formatting dates...")
gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# 3. Merge Weather + Power
# We group by time to get the Total Plant Output
df_gen = gen_data.groupby('DATE_TIME').sum(numeric_only=True)['DC_POWER'].reset_index()
df_weather = weather_data.groupby('DATE_TIME').mean(numeric_only=True)[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].reset_index()

df = pd.merge(df_gen, df_weather, on='DATE_TIME')

# 4. Filter (Remove night time data where sun is 0)
df = df[df['IRRADIATION'] > 0]

# 5. Train Model
print("Training Solar AI Model...")
# Inputs: Sun, Air Temp, Panel Temp
X = df[['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']]
# Target: Total Power
y = df['DC_POWER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save
with open('solar_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Success! 'solar_model.pkl' created.")