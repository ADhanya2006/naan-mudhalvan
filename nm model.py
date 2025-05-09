import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="EBPL Energy Optimization", layout="wide")

st.title("🔋 EBPL Energy Efficiency Optimization Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload energy usage CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocess
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Anomaly detection
    X_if = df[['Energy_Usage_kWh', 'Hour']]
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(X_if)
    df['Anomaly'] = df['Anomaly'].apply(lambda x: "Anomaly" if x == -1 else "Normal")

    # Demand prediction
    X = df[['DayOfWeek', 'Hour']]
    y = df['Energy_Usage_kWh']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_rf = RandomForestRegressor()
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Recommendation logic
    def recommend_action(usage, threshold=100):
        return "⚠️ Turn off idle machines" if usage > threshold else "✅ All good"
    df['Recommendation'] = df['Energy_Usage_kWh'].apply(recommend_action)

    # Display key metrics
    st.subheader("📊 Summary")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"{mae:.2f} kWh")
    col2.metric("Total Anomalies Detected", df['Anomaly'].value_counts().get('Anomaly', 0))

    # Line chart
    st.subheader("📈 Energy Usage Over Time")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Energy_Usage_kWh'], label='Energy Usage (kWh)', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Usage (kWh)")
    ax.legend()
    st.pyplot(fig)

    # Data table
    st.subheader("🧾 Annotated Data")
    st.dataframe(df[['Date', 'Energy_Usage_kWh', 'Recommendation', 'Anomaly']].reset_index(drop=True))

    # Download button
    st.download_button("Download Result CSV", df.to_csv(index=False), "energy_insights.csv", "text/csv")

else:
    st.info("Please upload a CSV file to begin analysis.")
