CoAP-Based Air Quality Monitoring and Random Forest Classification System
An end-to-end IoT and Machine Learning pipeline for real-time air quality monitoring in Kochi, India. The system fetches live pollutant metrics via the Open-Meteo REST API, serves them over the Constrained Application Protocol (CoAP / UDP), receives them via a CoAP client, classifies the AQI category using a Random Forest model trained on historical Indian air quality data (city_day.csv), and renders an interactive Streamlit dashboard.
1. System Architecture
+-----------------------------+
|    Open-Meteo REST API      |
|  (Kochi: 9.93°N, 76.27°E)   |
+-----------------------------+
               │ HTTP GET
               ▼
+-----------------------------+
|         CoAP Server         |
|   (aiocoap / Port 5683)     |
|   Resource: /airquality     |
+-----------------------------+
               │ CoAP GET (UDP, RFC 7252)
               ▼
+-----------------------------+
|         CoAP Client         |
|      (aiocoap client)       |
+-----------------------------+
               │ Raw JSON Metrics
               ▼
+-----------------------------+         +-------------------------------+
|     Streamlit Dashboard     | ──────> |     Random Forest Model       |
|  (Visualizes 5 Pollutants)  | <────── |   (Trained on city_day.csv)   |
+-----------------------------+         | Target: AQI_Bucket            |
                                        +-------------------------------+

2. Directory Structure
Create a root project directory named coap_aqi_system and organize your files as follows:
coap_aqi_system/
│
├── city_day.csv         # Kaggle historical dataset file (downloaded)
├── requirements.txt     # Python dependencies
├── train_model.py       # Script 1: Trains Random Forest model & saves .pkl
├── coap_server.py       # Script 2: CoAP server polling Open-Meteo API
├── coap_client.py       # Script 3: CoAP client fetching from CoAP server
└── dashboard.py         # Script 4: Streamlit UI + model inference

3. Prerequisites & Installation
Step 3.1: Python Environment Setup
Open a terminal in the project folder and verify Python 3.9+ is installed:
python --version

(Optional but recommended) Create and activate a virtual environment:
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate

Step 3.2: Create requirements.txt
Save the following content as requirements.txt:
aiocoap==0.4.7
requests>=2.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
streamlit>=1.28.0

Install all dependencies:
pip install -r requirements.txt

4. Dataset Setup
 * Download the dataset from Kaggle:
   * URL: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
 * Extract the archive and copy the file city_day.csv directly into the coap_aqi_system/ root directory.
5. Source Code Files
File 1: train_model.py
Trains a Random Forest classifier using features PM2.5, PM10, NO2, SO2, and O3 to predict AQI_Bucket (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
"""
train_model.py
Trains a Random Forest Classifier on historical city_day.csv data
and serializes the pipeline into rf_aqi_model.pkl.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_and_export():
    dataset_path = "city_day.csv"
    model_output_path = "rf_aqi_model.pkl"

    print("[Step 1] Loading historical dataset from city_day.csv...")
    df = pd.read_csv(dataset_path)

    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']
    target = 'AQI_Bucket'

    # Filter records with valid labels
    df = df.dropna(subset=[target])
    # Drop rows where all 5 pollutant features are missing
    df = df.dropna(subset=features, how='all')

    X = df[features]
    y = df[target]

    print(f"Total training samples: {len(X)}")
    print(f"Target distribution:\n{y.value_counts()}\n")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imputer + Random Forest Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(
            n_estimators=150,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("[Step 2] Training Random Forest model...")
    pipeline.fit(X_train, y_train)

    print("[Step 3] Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"[Step 4] Serializing model to '{model_output_path}'...")
    joblib.dump(pipeline, model_output_path)
    print("Model training complete.")

if __name__ == "__main__":
    train_and_export()

File 2: coap_server.py
Runs a CoAP server listening on UDP port 5683. When queried on /airquality, it fetches live weather/air quality parameters for Kochi from the Open-Meteo API and responds with a CoAP 2.05 CONTENT payload.
"""
coap_server.py
CoAP Server providing live air quality observations for Kochi via /airquality.
"""

import asyncio
import json
import urllib.request
import aiocoap
import aiocoap.resource as resource

API_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    "latitude=9.93&longitude=76.27&current=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone"
)

class AirQualityResource(resource.Resource):
    """Resource returning current air quality observations in JSON."""

    async def render_get(self, request):
        client_address = request.remote.hostinfo if request.remote else "unknown"
        print(f"[CoAP Server] Received GET request from {client_address}")

        try:
            req = urllib.request.Request(
                API_URL, 
                headers={'User-Agent': 'CoAP-AirQuality-Agent/1.0'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                api_payload = json.loads(response.read().decode('utf-8'))

            current = api_payload.get("current", {})

            # Format data to align with ML model features
            air_data = {
                "city": "Kochi",
                "latitude": 9.93,
                "longitude": 76.27,
                "timestamp": current.get("time"),
                "PM2.5": current.get("pm2_5"),
                "PM10": current.get("pm10"),
                "NO2": current.get("nitrogen_dioxide"),
                "SO2": current.get("sulphur_dioxide"),
                "O3": current.get("ozone")
            }

            payload_bytes = json.dumps(air_data).encode('utf-8')
            print(f"[CoAP Server] Responding with payload: {air_data}")
            return aiocoap.Message(payload=payload_bytes, code=aiocoap.numbers.codes.Code.CONTENT)

        except Exception as err:
            err_payload = json.dumps({"error": str(err)}).encode('utf-8')
            print(f"[CoAP Server Error] {err}")
            return aiocoap.Message(payload=err_payload, code=aiocoap.numbers.codes.Code.INTERNAL_SERVER_ERROR)

async def main():
    root = resource.Site()
    root.add_resource(['airquality'], AirQualityResource())

    # Listen on localhost, UDP port 5683
    await aiocoap.Context.create_server_context(root, bind=('127.0.0.1', 5683))
    print("[CoAP Server] Server listening at coap://127.0.0.1:5683/airquality")
    await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())

File 3: coap_client.py
A client module that sends a CoAP GET request to the local CoAP server and decodes the JSON response.
"""
coap_client.py
Asynchronous client sending CoAP GET requests to retrieve air quality data.
"""

import asyncio
import json
from aiocoap import Message, GET, Context

async def request_air_quality_coap(server_uri="coap://127.0.0.1:5683/airquality"):
    """Sends a CoAP GET request and returns parsed sensor readings."""
    context = await Context.create_client_context()
    request = Message(code=GET, uri=server_uri)
    
    try:
        response = await context.request(request).response
        payload_text = response.payload.decode('utf-8')
        return json.loads(payload_text)
    finally:
        await context.shutdown()

if __name__ == "__main__":
    print("[CoAP Client] Querying coap://127.0.0.1:5683/airquality ...")
    try:
        data = asyncio.run(request_air_quality_coap())
        print("[CoAP Client] Received Response:")
        print(json.dumps(data, indent=2))
    except Exception as exc:
        print(f"[CoAP Client Error] Could not connect to CoAP server: {exc}")

File 4: dashboard.py
Streamlit application that connects to the CoAP client, queries the CoAP server for Kochi's live metrics, feeds them to the loaded Random Forest model, and displays metrics and predictions.
"""
dashboard.py
Streamlit Dashboard for Kochi Air Quality Monitoring and Classification.
"""

import streamlit as st
import pandas as pd
import asyncio
import joblib
import os
from coap_client import request_air_quality_coap

st.set_page_config(
    page_title="Kochi Air Quality Monitoring",
    page_icon="🌿",
    layout="wide"
)

CATEGORY_COLORS = {
    "Good": "#2ecc71",
    "Satisfactory": "#27ae60",
    "Moderate": "#f39c12",
    "Poor": "#e67e22",
    "Very Poor": "#e74c3c",
    "Severe": "#8e44ad"
}

@st.cache_resource
def load_rf_model(model_path="rf_aqi_model.pkl"):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_rf_model()

st.title("🌿 CoAP-Based Air Quality Monitoring & Classification")
st.caption("City: **Kochi (Lat 9.93, Lon 76.27)** | Pipeline: Open-Meteo API ➔ CoAP Server ➔ CoAP Client ➔ Dashboard ➔ Random Forest Classifier")

if model is None:
    st.error("⚠️ Trained model 'rf_aqi_model.pkl' not found. Please run 'python train_model.py' first.")
    st.stop()

col_btn, _ = st.columns([1, 4])
with col_btn:
    poll_button = st.button("🔄 Poll CoAP Server", use_container_width=True)

# Fetch data via CoAP Client
coap_data = None
try:
    with st.spinner("Requesting live data via CoAP GET..."):
        coap_data = asyncio.run(request_air_quality_coap())
except Exception as error:
    st.error(f"CoAP Client Communication Failure: Make sure coap_server.py is running. Error: {error}")
    st.stop()

if coap_data and "error" not in coap_data:
    st.success(f"CoAP Transmission Verified (2.05 Content) | Timestamp: {coap_data.get('timestamp')}")
    st.divider()

    # Parameter Metrics
    st.subheader("📊 Live Air-Quality Parameters (Kochi)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(label="PM2.5 (µg/m³)", value=f"{coap_data.get('PM2.5', 0.0):.2f}")
    c2.metric(label="PM10 (µg/m³)", value=f"{coap_data.get('PM10', 0.0):.2f}")
    c3.metric(label="NO₂ (µg/m³)", value=f"{coap_data.get('NO2', 0.0):.2f}")
    c4.metric(label="SO₂ (µg/m³)", value=f"{coap_data.get('SO2', 0.0):.2f}")
    c5.metric(label="O₃ (µg/m³)", value=f"{coap_data.get('O3', 0.0):.2f}")

    # Build input feature frame for model
    feature_df = pd.DataFrame([{
        'PM2.5': coap_data.get('PM2.5'),
        'PM10': coap_data.get('PM10'),
        'NO2': coap_data.get('NO2'),
        'SO2': coap_data.get('SO2'),
        'O3': coap_data.get('O3')
    }])

    # Predict class
    predicted_bucket = model.predict(feature_df)[0]
    box_color = CATEGORY_COLORS.get(predicted_bucket, "#34495e")

    st.divider()
    st.subheader("🏷️ Random Forest AQI Classification")
    
    col_pred, col_prob = st.columns([1, 1])
    with col_pred:
        st.markdown(
            f"""
            <div style="background-color: {box_color}; padding: 30px; border-radius: 12px; text-align: center; color: white;">
                <h3 style="margin: 0; font-size: 1.2rem; font-weight: normal;">Predicted Air Quality Bucket</h3>
                <h1 style="margin: 10px 0 0 0; font-size: 2.8rem; font-weight: bold;">{predicted_bucket}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_prob:
        if hasattr(model.named_steps['classifier'], "predict_proba"):
            classes = model.named_steps['classifier'].classes_
            probabilities = model.predict_proba(feature_df)[0]
            prob_table = pd.DataFrame({"Category": classes, "Probability": probabilities}).sort_values(
                by="Probability", ascending=False
            )
            st.write("**Classification Confidence Breakdown:**")
            st.dataframe(prob_table.style.format({"Probability": "{:.2%}"}), use_container_width=True)
else:
    st.error(f"CoAP Server responded with error: {coap_data.get('error') if coap_data else 'No response'}")

6. Execution Order & Step-by-Step Instructions
Execute the system using three separate terminal windows in the specified order.
[Terminal 1] Train Model (Run once) ➔ Start CoAP Server
[Terminal 2] (Optional) Verify CoAP Client standalone
[Terminal 3] Launch Streamlit Dashboard

Step 6.1: Terminal 1 — Train the Random Forest Model
Ensure city_day.csv is present in the directory. Run:
python train_model.py

 * Expected Result: Preprocesses data, trains the Random Forest classifier, outputs test accuracy and classification metrics, and creates rf_aqi_model.pkl.
Step 6.2: Terminal 1 — Start the CoAP Server
In the same terminal (or a new one), start the CoAP server:
python coap_server.py

 * Expected Result:
   [CoAP Server] Server listening at coap://127.0.0.1:5683/airquality

 * Keep this terminal running.
Step 6.3: Terminal 2 — Test the CoAP Client (Verification)
Open a second terminal window and run:
python coap_client.py

 * Expected Result:
   [CoAP Client] Querying coap://127.0.0.1:5683/airquality ...
[CoAP Client] Received Response:
{
  "city": "Kochi",
  "latitude": 9.93,
  "longitude": 76.27,
  "timestamp": "2026-...",
  "PM2.5": 24.1,
  "PM10": 42.5,
  "NO2": 8.3,
  "SO2": 4.1,
  "O3": 18.2
}

 * In Terminal 1, verify that [CoAP Server] Received GET request appears.
Step 6.4: Terminal 3 — Launch the Dashboard
Open a third terminal window and execute:
streamlit run dashboard.py

 * Expected Result: Streamlit starts a local web server (defaults to http://localhost:8501).
 * Open your browser to http://localhost:8501.
 * Click the "🔄 Poll CoAP Server" button. The dashboard executes a live CoAP GET request to the local CoAP server, fetches Kochi's metrics from Open-Meteo, renders the 5 pollutant values, and displays the classified AQI category badge (Good, Satisfactory, Moderate, etc.) with model confidence probabilities.
7. Requirement Verification Matrix
| Requirement | Implementation Component | Verification Check |
|---|---|---|
| 1. CoAP Server & Client | coap_server.py & coap_client.py | Uses aiocoap binding UDP 5683 with resource path /airquality. |
| 2. Live Kochi Data Retrieval | coap_server.py via Open-Meteo API | Queries latitude=9.93&longitude=76.27 for Kochi upon receiving CoAP GET. |
| 3. CoAP GET Data Transfer | coap_client.py requesting coap://127.0.0.1:5683 | Returns standard 2.05 CONTENT with JSON payload containing all parameters. |
| 4. Parameter Dashboard | dashboard.py (Streamlit metrics) | Displays live cards for PM2.5, PM10, NO2, SO2, and O3. |
| 5. Random Forest Training | train_model.py using city_day.csv | Pipeline using SimpleImputer + RandomForestClassifier targeting AQI_Bucket. |
| 6. Predicted Classification Display | dashboard.py prediction box | Renders color-coded classification badge and confidence breakdown table. |
