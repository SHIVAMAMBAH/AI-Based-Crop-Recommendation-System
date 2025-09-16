import streamlit as st
import numpy as np
import joblib
import plotly.express as px

# Load model
model = joblib.load("classification_model.pkl")

# Categories in alphabetical order (LabelEncoder style)
class_labels = [
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee",
    "cotton", "grapes", "jute", "kidneybeans", "lentil", "maize",
    "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya",
    "pigeonpeas", "pomegranate", "rice", "watermelon"
]

input_features = ["Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "Ph", "Rainfall"]


st.set_page_config(
    page_title="Cropper",
    page_icon="🌱",                     
)

st.title("🌱 Crop Recommendation System")
st.write("Welcome to Cropper, a crop recommendation system which will help you in choosing the right crop for your fields.")
st.markdown("---")
st.write("I considers the following factors and let you choose the crop on the basis of those factors.")
st.write("Just enter the following data and press Predict button and you will have your crop predicted!")
st.markdown("---")

# Input fields
features = []
cols = st.columns(4) 

for i in range(7):
    col = cols[i % 4]   # pick column based on index
    with col:
        val = st.number_input(input_features[i], value=0.0)
        features.append(val)

features = np.array(features).reshape(1, -1)

# Prediction
if st.button("Predict"):
    probs = model.predict_proba(features)[0]   # probabilities for each class
    predicted_class = int(np.argmax(probs))
    predicted_label = class_labels[predicted_class]

    st.success(f"✅ Predicted Crop: **{predicted_label}**")

    # Create DataFrame for Plotly
    import pandas as pd
    df = pd.DataFrame({
        "Crop": class_labels,
        "Probability": probs
    })

    # Sort by probability for better visualization
    df = df.sort_values("Probability", ascending=False)

    # Plot with animation
    fig = px.bar(
        df,
        x="Probability",
        y="Crop",
        orientation="h",
        title="Prediction Probabilities",
        text="Probability",
        color="Crop",
        animation_frame=None,  # could simulate multiple steps if you want
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, transition_duration=500)

    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.write("Made by Shivam Sharma")

# SideBar

crop_info = {
    "Rice": {
        "Scientific Name": "Oryza sativa",
        "Optimal Temperature": "20-35°C",
        "Water Requirement": "High"
    },
    "Wheat": {
        "Scientific Name": "Triticum aestivum",
        "Optimal Temperature": "10-25°C",
        "Water Requirement": "Moderate"
    },
    "Maize": {
        "Scientific Name": "Zea mays",
        "Optimal Temperature": "18-27°C",
        "Water Requirement": "Moderate"
    }
}

st.sidebar.title("Cropper")
st.sidebar.write("Cropper Helps You in selecting the right crop for your fields.")
st.sidebar.markdown("---")
st.sidebar.write("Select a crop from to know more about it.")

# Sidebar input widgets
option = st.sidebar.selectbox("Select a crop:", class_labels)


if option:
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Crop Name:** {option}")
    info = crop_info[option]
    for key, value in info.items():
        st.sidebar.write(f"**{key}:** {value}")