import streamlit as st
import json
import pandas as pd
import base64
from modules.story_generator import generate_story

# Set page configuration
st.set_page_config(page_title="Accident Story Generator", layout="wide")

# Set background image
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: #fff;
            }}
            .block-container {{
                background-color: rgba(0, 0, 0, 0.6);
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.5);
            }}
            .stTextArea textarea {{
                background-color: #fefefe;
                color: #333;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Background image not loaded: {e}")

set_background("C:/Users/BIT/OneDrive/Desktop/accident_story_generator/tata.jpg")

# Title
st.markdown("<h1 style='text-align: center; color: white;'> AutoAssist AI: Accident Story Generator</h1>", unsafe_allow_html=True)
st.markdown("<h6 style=' text-align: center; color: white;'> An AI-powered tool that automatically generates detailed accident narratives from car data like speed, location, and driver input for legal and insurance use.</h6>", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Driver input
with st.chat_message("user"):
    driver_notes = st.text_area("📝 Driver Notes", "Applied brakes but visibility was poor...")

# Load CSV and JSON safely
try:
    speed_df = pd.read_csv("data/speed_logs.csv")
    gps_data = json.load(open("data/gps_data.json"))
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Select conditions
col1, col2 = st.columns(2)
with col1:
    road_condition = st.selectbox("🛣️ Road Condition", ["Dry", "Rainy", "Icy"])
with col2:
    weather_condition = st.selectbox("Weather Condition", ["Sunny", "Foggy", "Rainy"])


# Optional raw data display
with st.expander("📊 View Raw Data"):
    st.subheader("Speed Logs")
    st.dataframe(speed_df)

    st.subheader("GPS Data")
    st.json(gps_data)

# Speed Over Time using Streamlit line_chart
st.subheader("📈 Speed Over Time")
try:
    chart_df = speed_df.set_index("timestamp")
    st.line_chart(chart_df["speed"])
except Exception as e:
    st.warning(f"Speed graph error: {e}")



# Generate story
if st.button(" Generate Accident Story"):
    with st.spinner("Generating story..."):
        with st.chat_message("user"):
            st.markdown(f"**Driver's Note:** {driver_notes}")

        try:
            story = generate_story(speed_df.to_dict(), gps_data, driver_notes, road_condition, weather_condition)

            # Store and display story
            st.session_state.chat_history.append(("user", driver_notes))
            st.session_state.chat_history.append(("assistant", story))

        except Exception as e:
            st.error(f"❌ Error during story generation: {e}")

# Render conversation
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
