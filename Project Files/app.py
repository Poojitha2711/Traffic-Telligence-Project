import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

st.set_page_config(page_title="TrafficTelligence - Trip Forecast", layout="centered")
st.title("🚦 TrafficTelligence - AP Trip Duration Forecast")

# Upload and load dataset
uploaded_file = st.file_uploader("Upload your cleaned_green_tripdata_2025_03.csv file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='ascii')

    # Show dataset
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Data Visualizations
    st.subheader("📌 Data Insights and Visualizations")

    # Trip Duration Distribution
    st.markdown("#### ⏱️ Trip Duration Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['trip_duration'], bins=30, kde=True, ax=ax1, color="skyblue")
    ax1.set_xlabel("Trip Duration (minutes)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Trip Durations")
    st.pyplot(fig1)

    # Trip Distance vs Duration Scatter
    st.markdown("#### 🛣️ Trip Distance vs Trip Duration")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='trip_distance', y='trip_duration', hue='is_rush_hour', palette='coolwarm', ax=ax2)
    ax2.set_xlabel("Trip Distance (km)")
    ax2.set_ylabel("Trip Duration (minutes)")
    ax2.set_title("Trip Distance vs Duration (Colored by Rush Hour)")
    st.pyplot(fig2)

    # Average Duration by Hour
    st.markdown("#### 🕘 Average Trip Duration by Pickup Hour")
    avg_duration_by_hour = df.groupby('pickup_hour')['trip_duration'].mean().reset_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(data=avg_duration_by_hour, x='pickup_hour', y='trip_duration', palette="viridis", ax=ax3)
    ax3.set_xlabel("Pickup Hour")
    ax3.set_ylabel("Average Duration (minutes)")
    ax3.set_title("Hourly Trip Duration Patterns")
    st.pyplot(fig3)

    # Weekend vs Weekday Comparison
    st.markdown("#### 📅 Weekend vs Weekday Trip Duration")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df, x='is_weekend', y='trip_duration', palette='pastel', ax=ax4)
    ax4.set_xticklabels(['Weekday', 'Weekend'])
    ax4.set_xlabel("Day Type")
    ax4.set_ylabel("Trip Duration (minutes)")
    ax4.set_title("Trip Duration Comparison: Weekend vs Weekday")
    st.pyplot(fig4)

    # Features and Target
    features = ['trip_distance', 'pickup_hour', 'is_weekend', 'is_rush_hour']
    target = 'trip_duration'

    if all(col in df.columns for col in features + [target]):
        X = df[features]
        Y = df[target]

        # Train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # Predict and evaluate
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        st.subheader("📈 Model Evaluation")
        st.write(f"**Mean Squared Error:** {mse:.2f}")
        st.write(f"**R-squared (R²):** {r2:.4f}")

        # Prediction input form
        st.subheader("🚗 Trip Prediction Input")

        col1, col2 = st.columns(2)
        with col1:
            source = st.text_input("Source", value="Vijayawada")
        with col2:
            destination = st.text_input("Destination", value="Guntur")

        vehicle = st.selectbox("Vehicle Type", ["Car", "Bike", "Bus", "Truck"])
        trip_distance = st.number_input("Estimated Trip Distance (km)", value=40.0, min_value=0.0)
        pickup_hour = st.slider("Pickup Hour", 0, 23, 9)
        is_weekend = st.radio("Is it a Weekend?", [0, 1], horizontal=True)
        is_rush_hour = st.radio("Is it Rush Hour?", [0, 1], horizontal=True)

        if st.button("🔍 Predict Duration"):
            input_data = pd.DataFrame([[trip_distance, pickup_hour, is_weekend, is_rush_hour]], columns=features)
            predicted_duration = model.predict(input_data)[0]

            # Adjust time based on vehicle type
            if vehicle.lower() == "bike":
                predicted_duration *= 0.8
            elif vehicle.lower() == "bus":
                predicted_duration *= 1.2
            elif vehicle.lower() == "truck":
                predicted_duration *= 1.5

            traffic_level = "High" if predicted_duration > 90 else "Moderate" if predicted_duration > 45 else "Low"
            traffic_volume = int(50 + predicted_duration * 2 + 30)

            # Result display
            st.markdown("### 🧾 Trip Summary")
            st.write(f"**Date & Time:** {datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}")
            st.write(f"**From:** {source} → **To:** {destination}")
            st.write(f"**Distance:** {trip_distance} km")
            st.write(f"**Estimated Time:** {predicted_duration:.1f} minutes")
            st.write(f"**Traffic Level:** {traffic_level}")
            st.write(f"**Estimated Vehicle Volume:** {traffic_volume} vehicles")

    else:
        st.error("Required columns not found in uploaded dataset.")
else:
    st.info("Please upload the cleaned_green_tripdata_2025_03.csv file to proceed.")