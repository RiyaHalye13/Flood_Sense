import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df = pd.read_csv("data/flood.csv")
x=df.iloc[:,0:20]
y = df["FloodProbability"]
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
st.title("Flood Risk Factors Input Form")

st.subheader("Enter Scores for the Following Factors (0 to 10):")

monsoon_intensity = st.number_input("Monsoon Intensity", min_value=0.0, max_value=10.0)
topography_drainage = st.number_input("Topography Drainage", min_value=0.0, max_value=10.0)
river_management = st.number_input("River Management Score", min_value=0.0, max_value=10.0)
deforestation = st.number_input("Deforestation Level", min_value=0.0, max_value=10.0)
urbanization = st.number_input("Urbanization Score", min_value=0.0, max_value=10.0)
climate_change = st.number_input("Climate Change Impact Score", min_value=0.0, max_value=10.0)
dams_quality = st.number_input("Dams Quality Score", min_value=0.0, max_value=10.0)
siltation = st.number_input("Siltation Level", min_value=0.0, max_value=10.0)
agricultural_practices = st.number_input("Agricultural Practices Score", min_value=0.0, max_value=10.0)
ineffective_disaster_preparedness = st.number_input("Ineffective Disaster Preparedness Score", min_value=0.0, max_value=10.0)
drainage_systems = st.number_input("Drainage Systems Score", min_value=0.0, max_value=10.0)
landslides = st.number_input("Landslide Risk Score", min_value=0.0, max_value=10.0)
watersheds = st.number_input("Watershed Condition Score", min_value=0.0, max_value=10.0)
deteriorating_infrastructure = st.number_input("Deteriorating Infrastructure Score", min_value=0.0, max_value=10.0)
population_score = st.number_input("Population Score", min_value=0.0, max_value=10.0)
wetland_loss = st.number_input("Wetland Loss Score", min_value=0.0, max_value=10.0)
inadequate_planning = st.number_input("Inadequate Planning Score", min_value=0.0, max_value=10.0)
political_factors = st.number_input("Political Factors Score", min_value=0.0, max_value=10.0)
encroachments = st.number_input("Encroachments Score", min_value=0.0, max_value=10.0)
coastal_vulnerability = st.number_input("Coastal Vulnerability Score", min_value=0.0, max_value=10.0)

if st.button("Submit"):
    input_data = [[
        monsoon_intensity, topography_drainage, river_management, deforestation, urbanization,
        climate_change, dams_quality, siltation, agricultural_practices, ineffective_disaster_preparedness,
        drainage_systems, landslides, watersheds, deteriorating_infrastructure, population_score,
        wetland_loss, inadequate_planning, political_factors, encroachments, coastal_vulnerability
    ]]

    # Predict
    result = model.predict(input_data)[0]  # Extract scalar from array

    # Display message based on result
    if 0 <= result < 0.4:
        st.success("Low chances of flood 🌤️")
    elif 0.4 <= result < 0.75:
        st.warning("Mild chances of flood 🌦️")
    elif 0.75 <= result < 1:
        st.error("High chances of flood ⚠️")
    elif 1 <= result:
        st.error("Very high chances of flood 🚨")

