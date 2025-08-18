# -KPM-Predictor-xApp-By-eng.ahmed-alsubaihy
 KPM Predictor xApp is an open-source project for O-RAN that collects real-time Key Performance Measurements (KPM), predicts Physical Resource Block (PRB) usage using a trained LSTM model, and applies RAN control decisions dynamically.  This xApp is designed to run in an O-RAN near-RT RIC setup and integrates with InfluxDB for measurement storage
Features

Connects to InfluxDB to fetch live UE measurement data.

Supports mapping raw KPM field names to human-readable feature names.

Uses a trained LSTM deep learning model to predict PRB usage.

Implements real-time visualization of actual vs. predicted PRB values.

Provides RAN control logic to dynamically adjust PRB allocation for UEs.

Modular class design (KpmPredictorXapp) extending xAppBase for easy integration.

🗂 Project Structure

Configuration Section
Sets InfluxDB credentials, model/scaler paths, feature lists, and logging.

Helper Functions

fetch_latest_data(ue_id): Queries InfluxDB for the latest UE metrics.

preprocess_data(df, scaler, features): Cleans, fills missing values, and scales data.

make_prediction(model, scaler, sequence, features): Runs prediction and inverses scaling.

KpmPredictorXapp Class

Handles subscriptions to KPM reports from the E2 node.

Stores incoming KPM data into InfluxDB.

Runs predictions and updates plots.

Implements RAN control logic via RanControl().

Main Block
Provides a CLI (argparse) to configure the app:

python3 kpm_merged.py \
    --config config.json \
    --http_server_port 8092 \
    --rmr_port 4560 \
    --e2_node_id gnbd_001_001_00019b_0 \
    --ran_func_id 2 \
    --kpm_report_style 2 \
    --ue_ids 0 \
    --metrics DRB.UEThpDl

🔮 Prediction Pipeline

Subscribe to KPM reports from E2 node via E2SM-KPM.

Write incoming data into InfluxDB.

Fetch latest measurements for a given UE.

Preprocess features (scaling, missing values).

Run LSTM prediction for PRB usage (1 second ahead).

Visualize results (actual vs predicted PRB).

Apply RAN control based on predictions and thresholds.

📊 Visualization

The xApp generates live plots for:

Actual vs Predicted PRB usage.

Differences from a fixed PRB reference (50).

UE-level PRB allocation decisions.

🛠 Requirements

Python 3.8+

TensorFlow / Keras

Scikit-learn

Pandas / Numpy

InfluxDB client

Matplotlib

Future Improvements

Replace placeholder random PRB values with actual labels.

Optimize LSTM input shape (currently uses single-step).

Expand RAN control strategies (multi-UE scheduling).

Containerize xApp with Docker for deployment in O-RAN testbed.


created BY 
ahmed alsubaihy 
waleed alrobian 
