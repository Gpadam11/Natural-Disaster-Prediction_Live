import streamlit as st
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz

# Load trained earthquake model
earthquake_model = joblib.load("earthquake_model_25.pkl")

# Load world cities data
@st.cache_data
def load_city_data():
    return pd.read_csv("worldcities.csv", encoding="latin1")[["country", "city", "lat", "lng"]]

cities_df = load_city_data()

# Fetch last 100 earthquakes for a selected location and compute average depth within 300km range
def fetch_avg_depth(lat, lng):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=90)
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 3.0,
        "maxradiuskm": 300,
        "latitude": lat,
        "longitude": lng,
    }
    response = requests.get(url, params=params)
    data = response.json()
    features = data.get("features", [])
    depths = [
        feature["geometry"]["coordinates"][2]
        for feature in features[:100]
        if feature["geometry"]["coordinates"][2] is not None
    ]
    return sum(depths) / len(depths) if depths else None

# Fetch recent earthquake data
def fetch_live_earthquake_data(lat, lng):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=90)
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 3.0,
        "maxradiuskm": 300,
        "latitude": lat,
        "longitude": lng,
    }
    response = requests.get(url, params=params)
    data = response.json()
    features = data.get("features", [])
    earthquakes = [
        {
            "Time": datetime.utcfromtimestamp(feature["properties"]["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
            "Magnitude": feature["properties"]["mag"],
            "Depth (km)": feature["geometry"]["coordinates"][2],
            "Place": feature["properties"]["place"],
        }
        for feature in features
    ]
    return pd.DataFrame(earthquakes) if earthquakes else pd.DataFrame(
        columns=["Time", "Magnitude", "Depth (km)", "Place"])

# Streamlit UI
st.title("🌍 Live Earthquake Risk Predictor")

# Sidebar inputs
selected_country = st.sidebar.selectbox("Select Country", ["Select a country"] + sorted(cities_df["country"].unique()))

if selected_country != "Select a country":
    filtered_cities = cities_df[cities_df["country"] == selected_country]
    selected_city = st.sidebar.selectbox("Select City", ["Select a city"] + sorted(filtered_cities["city"].unique()))

    if selected_city != "Select a city":
        city_info = filtered_cities[filtered_cities["city"] == selected_city].iloc[0]
        latitude, longitude = city_info["lat"], city_info["lng"]
        avg_depth = fetch_avg_depth(latitude, longitude)

        st.sidebar.write(f"📍 **Latitude:** {latitude:.6f}, **Longitude:** {longitude:.6f}")
        if avg_depth is not None:
            st.sidebar.write(f"🔽 **Avg Depth (300km Range):** {avg_depth:.2f} km")

        if st.sidebar.button("Predict"):
            input_data = pd.DataFrame([[avg_depth if avg_depth else 10, latitude, longitude]],
                                      columns=["Depth (km)", "Latitude", "Longitude"])
            magnitude = earthquake_model.predict(input_data)[0]
            probability = min(max((magnitude - 3) / 7, 0), 1) * 100

            if magnitude < 4.5:
                st.success("🟢 **Area is SAFE**")
            elif 4.5 <= magnitude < 6.0:
                st.warning("🟡 **Moderate Risk**")
            else:
                st.error("🔴 **DANGER! High Risk**")

            st.success(f"🌍 **Predicted Magnitude:** {magnitude:.2f}")
            st.info(f"⚠️ **Earthquake Probability:** {probability:.2f}%")

            st.header(f"📊 Recent Earthquakes near {selected_city}, {selected_country}")
            live_data = fetch_live_earthquake_data(latitude, longitude)
            if not live_data.empty:
                st.write(f"📅 **Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
                st.dataframe(live_data.head(10))
            else:
                st.write("✅ No significant earthquakes detected in the past 3 months.")