
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model, scaler and metadata
model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')
metadata = joblib.load('metadata.joblib')

features = metadata['features']
remaining_sensors = metadata['remaining_sensors']

# App title
st.set_page_config(page_title="Jet Engine RUL Predictor", layout="wide")
st.title("✈️ Predictive Maintenance: Jet Engine RUL Predictor")
st.markdown("""
Predict the **Remaining Useful Life (RUL)** of a NASA CMAPSS Turbofan engine based on sensor readings.
""")

# Load data for display
@st.cache_data
def load_data():
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    test_df = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    y_test = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    return test_df, y_test

test_df, y_test = load_data()

# Sidebar for selection
st.sidebar.header("Engine Selection")
engine_id = st.sidebar.selectbox("Select Engine Unit ID", test_df['unit_nr'].unique())

# Filter data for selected engine
engine_data = test_df[test_df['unit_nr'] == engine_id].copy()

# Feature Engineering for the app
def add_rolling_features(df, sensors, window=5):
    for s in sensors:
        df[s + '_rolling_mean'] = df[s].rolling(window).mean()
        df[s + '_rolling_std'] = df[s].rolling(window).std()
    return df.fillna(0)

engine_data_processed = add_rolling_features(engine_data, remaining_sensors)
engine_data_processed[features] = scaler.transform(engine_data_processed[features])

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Engine Stats")
    last_cycle = engine_data['time_cycles'].max()
    st.metric("Total Cycles Flown", last_cycle)
    
    # Prediction
    last_row = engine_data_processed.iloc[-1:][features]
    prediction = model.predict(last_row)[0]
    actual = y_test.iloc[int(engine_id)-1]['RUL']
    
    st.subheader("RUL Prediction")
    st.success(f"Estimated Remaining Life: **{int(prediction)} cycles**")
    st.info(f"Actual Remaining Life (Ground Truth): **{int(actual)} cycles**")
    st.warning(f"Error Margin: {abs(int(prediction) - int(actual))} cycles")

with col2:
    st.subheader("Sensor Health Trends")
    sensor_to_plot = st.selectbox("Choose Sensor to Visualize", remaining_sensors)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=engine_data['time_cycles'], y=engine_data[sensor_to_plot], ax=ax, color='blue', label='Raw Signal')
    # Rolling mean for trend
    rolling_mean = engine_data[sensor_to_plot].rolling(window=10).mean()
    sns.lineplot(x=engine_data['time_cycles'], y=rolling_mean, ax=ax, color='red', label='Trend (Rolling Mean)')
    
    ax.set_title(f"Sensor {sensor_to_plot} over Time")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Sensor Value")
    st.pyplot(fig)

# Show data table
if st.checkbox("Show Raw Sensor Data"):
    st.write(engine_data.tail(10))

st.markdown("---")
st.caption("Developed for ML Lab Project - Jet Engine Predictive Maintenance")
