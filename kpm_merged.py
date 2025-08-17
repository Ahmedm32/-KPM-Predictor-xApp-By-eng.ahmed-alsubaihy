#!/usr/bin/env python3

import argparse
import signal
import time
import pandas as pd
import numpy as np
import joblib
import collections
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from lib.xAppBase import xAppBase

# ----------------------------- Configuration -----------------------------

# InfluxDB Configuration
Plot_BUCKET = 'prb_trial_4'
BUCKET = "kpmxapp"
ORG = "oranlab"
TOKEN = "MvhlvxyWU_IDZRvkzjDHpP125QNFAFVEaeAwwB7fv5RosYoOleAgTXIglFKCcUfKA6bijA_FrxqObILYBtqLpA=="
URL = "http://10.0.2.21:8086"

# Prediction Configuration
FETCH_INTERVAL = 1  # in seconds
SEQUENCE_LENGTH = 1
PREDICTION_OFFSET = 1  # Number of steps ahead to predict

# Feature Configuration
FEATURES = [
    "dl_mcs", "dl_buffer [bytes]", "tx_brate downlink [Mbps]", "tx_pkts downlink",
    "tx_errors downlink (%)", "dl_cqi", "ul_mcs", "ul_buffer [bytes]",
    "rx_brate uplink [Mbps]", "rx_pkts uplink", "rx_errors uplink (%)",
    "ul_sinr", "phr", "sum_requested_prbs"
]
TARGET = "sum_requested_prbs"

FEATURE_NAME_MAPPING = {
    'DRB.UEThpDl': 'tx_brate downlink [Mbps]',
    'DRB.UEThpUl': 'rx_brate uplink [Mbps]',
    'RRU.PrbAvailDl': 'sum_requested_prbs',
    'dl.cqi': 'dl_cqi',
    'ul.sinr': 'ul_sinr',
    'dl.mcs': 'dl_mcs',
    'dl.BufferBytes': 'dl_buffer [bytes]',
    'tx.PktsDownlink': 'tx_pkts downlink',
    'ul.mcs': 'ul_mcs',
    'ul.BufferBytes': 'ul_buffer [bytes]',
    'rx.PktsUplink': 'rx_pkts uplink',
    'rx.PktsUplinkErPer': 'rx_errors uplink (%)',
    'tx.PktsUplinkErPer': 'tx_errors downlink (%)',
    'phr': 'phr'
}

# File Paths
# MODEL_PATH = './trained_rf_model.pkl' # trained_model1.h5
# SCALER_PATH = './scaler2.pkl'
# LOG_FILE = 'predictor_logs.log'

MODEL_PATH = './trained_lstm_model_optimized.h5' # trained_model1.h5
SCALER_PATH = './scaler_lstm_X_optimized.pkl'
LOG_FILE = 'predictor_logs.log'

# ----------------------------- Setup Logging -----------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# ----------------------------- Load Model and Scaler -----------------------------

try:
    model =  joblib.load(MODEL_PATH) #load_model(MODEL_PATH, compile=False)
    logging.info("Successfully loaded the LSTM model from %s.", MODEL_PATH)
except Exception as e:
    logging.error("Error loading the LSTM model: %s", e)
    raise

try:
    scaler = joblib.load(SCALER_PATH)
    logging.info("Successfully loaded the scaler from %s.", SCALER_PATH)
except Exception as e:
    logging.error("Error loading the scaler: %s", e)
    raise

# ----------------------------- Connect to InfluxDB -----------------------------

try:
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    query_api = client.query_api()
    write_api = client.write_api(write_options=SYNCHRONOUS)
    logging.info("Successfully connected to InfluxDB.")
except Exception as e:
    logging.error("Error connecting to InfluxDB: %s", e)
    raise

# ----------------------------- Helper Functions -----------------------------

def fetch_latest_data(ue_id: str):
    """Fetch the latest data from InfluxDB using Flux query."""
    query = f'''
        from(bucket: "{BUCKET}")
        |> range(start: -10m)
        |> filter(fn: (r) => r["_measurement"] == "ue_info")
        |> filter(fn: (r) => r["ue_id"] == "{ue_id}")
        |> filter(fn: (r) => r["_field"] == "dl.cqi" or 
                  r["_field"] == "ul.sinr" or 
                  r["_field"] == "dl.mcs" or 
                  r["_field"] == "DRB.UEThpDl" or 
                  r["_field"] == "DRB.UEThpUl" or 
                  r["_field"] == "ul.mcs" or 
                  r["_field"] == "RRU.PrbAvailDl" or 
                  r["_field"] == "dl.BufferBytes" or 
                  r["_field"] == "ul.BufferBytes" or 
                  r["_field"] == "tx.PktsDownlink" or 
                  r["_field"] == "rx.PktsUplink" or 
                  r["_field"] == "rx.PktsUplinkErPer" or
                  r["_field"] == "tx.PktsUplinkErPer" or  
                  r["_field"] == "phr")
        |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
        |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
    '''
    try:
        result = query_api.query(query=query)
        metrics_data = []
        for table in result:
            for record in table.records:
                metrics_data.append(record.values)
        df = pd.DataFrame(metrics_data)
        logging.info("Fetched latest data successfully.")
        df.rename(columns=FEATURE_NAME_MAPPING, inplace=True)
        return df.tail(1)  #df.tail(10) before
    except Exception as e:
        logging.error("Error fetching data from InfluxDB: %s", e)
        return pd.DataFrame()

def preprocess_data(df, scaler, features):
    """Preprocess the data: select features, handle missing values, scale."""
    try:
        data = df[features].copy()
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.fillna(0, inplace=True)
        data['sum_requested_prbs']=np.random.normal(50, 20, size=len(data))
        print(data)
        scaled_data = scaler.fit_transform(data) #scaler.fit_transform(data)
        print('this var')
        print(scaled_data)
        return scaled_data
    except Exception as e:
        logging.error("Error during preprocessing: %s", e)
        return None

def make_prediction(model, scaler, sequence, features):
    """Make a prediction using the model based on the current sequence."""
    try:
        input_data = sequence #np.array(sequence).reshape((1, SEQUENCE_LENGTH, len(features)))
        print(input_data)
        pred_scaled = model.predict(input_data)
        dummy = np.zeros((1, len(features)))
        dummy[0, features.index(TARGET)] = pred_scaled
        print(dummy)
        pred_inversed = scaler.inverse_transform(dummy)
        pred = pred_inversed[0, features.index(TARGET)]
        return pred
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return None

# ----------------------------- xApp Class -----------------------------

class KpmPredictorXapp(xAppBase):
    def __init__(self, config, http_server_port, rmr_port):
        super(KpmPredictorXapp, self).__init__(config, http_server_port, rmr_port)
        self.sequence = []
        self.actual_list = []
        self.predicted_list = []
        self.predictions_buffer = collections.deque()
        
        # Setup interactive plots
        plt.ion()
        self.fig1, self.ax1 = plt.subplots(figsize=(12, 6))
        self.line_actual, = self.ax1.plot([], [], marker='o', linestyle='-', label='Actual PRB')
        self.line_pred, = self.ax1.plot([], [], marker='x', linestyle='--', label='Predicted PRB')
        self.ax1.set_title('Actual vs. Predicted PRB (1 Second Ahead)')
        self.ax1.set_xlabel('Measurement Index')
        self.ax1.set_ylabel('PRB Value')
        self.ax1.legend()
        self.ax1.grid(True)

        self.fig2, self.ax2 = plt.subplots(figsize=(12, 6))
        self.line_actual2, = self.ax2.plot([], [], marker='o', linestyle='-', label='Actual PRB Difference')
        self.ax2.set_title('Difference between Actual PRB and Fixed PRB (50)')
        self.ax2.set_xlabel('Measurement Index')
        self.ax2.set_ylabel('PRB Value Difference')
        self.ax2.legend()
        self.ax2.grid(True)

        self.fig3, self.ax3 = plt.subplots(figsize=(12, 6))
        self.line_pred2, = self.ax3.plot([], [], marker='x', linestyle='--', label='Predicted PRB Difference')
        self.ax3.set_title('Difference between Predicted PRB and Fixed PRB (50)')
        self.ax3.set_xlabel('Measurement Index')
        self.ax3.set_ylabel('PRB Value Difference')
        self.ax3.legend()
        self.ax3.grid(True)


    def RanControl(self,e2_node_id,ue,ded,m,per):
        self.e2sm_rc.control_slice_level_prb_quota(e2_node_id, ue, m, per, dedicated_prb_ratio=ded, ack_request=1)            
        return None

    def my_subscription_callback(self, e2_agent_id, subscription_id, indication_hdr, indication_msg, kpm_report_style, ue_id):
        """Process incoming KPM data and store in InfluxDB."""
        try:
            # Print received data
            if kpm_report_style == 2:
                print(f"\nRIC Indication Received from {e2_agent_id} for Subscription ID: {subscription_id}, KPM Report Style: {kpm_report_style}, UE ID: {ue_id}")
            else:
                print(f"\nRIC Indication Received from {e2_agent_id} for Subscription ID: {subscription_id}, KPM Report Style: {kpm_report_style}")

            # Extract header and measurement data
            indication_hdr = self.e2sm_kpm.extract_hdr_info(indication_hdr)
            meas_data = self.e2sm_kpm.extract_meas_data(indication_msg)
            print("E2SM_KPM RIC Indication Content:")
            print(f"-ColletStartTime: {indication_hdr['colletStartTime']}")
            print("-Measurements Data:")
            granulPeriod = meas_data.get("granulPeriod", None)
            if granulPeriod is not None:
                print(f"-granulPeriod: {granulPeriod}")

            # Write to InfluxDB
            data = Point("ue_info").tag("ue_id", ue_id)
            if kpm_report_style in [1, 2]:
                for metric_name, value in meas_data["measData"].items():
                    print(f"--Metric: {metric_name}, Value: {value}")
                    data.field(metric_name, value[0])
                write_api.write(bucket=BUCKET, org=ORG, record=data)
                #self.predict_and_visualize()
                
                prb_value = meas_data["measData"].get("RRU.PrbAvailDl", [None])[0]
                buff = meas_data["measData"].get("dl.BufferBytes", [None])[0]
                predicted_var1 = self.predict_and_visualize(ue_id)
                predicted_var = meas_data["measData"].get("DRB.UEThpDl", [None])[0]

                #print("xxxxxxxxxxxxxxxxxxxx")
                #print("prb_value:")
                #print(prb_value)
                print("predicted_var:")
                #print(predicted_var)
                print(predicted_var1)
                print("xxxxxxxxxxxxxxxxxxxx")
                prb=104
                low_prb=31
                high_prb=prb-low_prb

                if ue_id==10:
                    prb_value_ue_h = meas_data["measData"].get("RRU.PrbAvailDl", [None])[0]
                    ue_c=0
                    # if (int(abs(predicted_var))>49 and buff >100000) or (int(abs(predicted_var)))>60:
                    #     self.RanControl(e2_node_id,ue_c,1,low_prb)
                    #     self.RanControl(e2_node_id,ue_id,high_prb,100)

                    if (int(abs(prb_value_ue_h))>49 and buff >100000) or (int(abs(prb_value_ue_h)))>60:
                        self.RanControl(e2_node_id,ue_c,1,low_prb)
                        self.RanControl(e2_node_id,ue_id,high_prb,100)

                    else:
                        self.RanControl(e2_node_id,ue_c,1,100)
                        self.RanControl(e2_node_id,ue_id,1,100)

                # if ue_id==5:
                #    prb_value_ue_h = meas_data["measData"].get("RRU.PrbAvailDl", [None])[0]
                #    down_value_ue_h = meas_data["measData"].get("DRB.UEThpDl", [None])[0]
                #    print(down_value_ue_h)
                #    ue_c=0
                #     # if (int(abs(predicted_var))>49 and buff >100000) or (int(abs(predicted_var)))>60:
                #     #     self.RanControl(e2_node_id,ue_c,1,low_prb)
                #     #     self.RanControl(e2_node_id,ue_id,high_prb,100)

                # if (int(abs(prb_value_ue_h))>0):
                #         self.RanControl(e2_node_id,ue_c,0,10,80)
                #         self.RanControl(e2_node_id,ue_id,20,50,100)

                # if down_value_ue_h==None:
                #         print("##########")
                #         self.RanControl(e2_node_id,ue_c,0,1,100)
                #         #self.RanControl(e2_node_id,ue_id,1,100)

                
                

                prb_point = (
                    Point("prb_info")
                    .tag("ue_id", ue_id)
                    .field("PRB", prb_value)
                   # .field("predicted_metric", predicted_var)
                    .field("downlink", predicted_var)
                            )
                write_api.write(bucket=Plot_BUCKET, org=ORG, record=prb_point)
                    #time.sleep(FETCH_INTERVAL)
            else:
                for ue_id, ue_meas_data in meas_data["ueMeasData"].items():
                    print(f"--UE_id: {ue_id}")
                    data = Point("ue_info").tag("ue_id", ue_id)
                    granulPeriod = ue_meas_data.get("granulPeriod", None)
                   # if granulPeriod is not None:
                        #print(f"---granulPeriod: {granulPeriod}")
                    for metric_name, value in ue_meas_data["measData"].items():
                        #print(f"---Metric: {metric_name}, Value: {value}")
                        data.field(metric_name, value[0])
                    write_api.write(bucket=BUCKET, org=ORG, record=data)
                    print(f"Data written at {time.strftime('%H:%M:%S')}")


        except Exception as e:
            logging.error("Error in subscription callback: %s", e)



    def predict_and_visualize(self,u):
        """Fetch data, make predictions, and update visualizations."""
        try:
            current_time = time.time()
            # Check scheduled predictions
            while self.predictions_buffer and self.predictions_buffer[0][0] <= current_time:
                due_time, pred_value = self.predictions_buffer.popleft()
                df_latest = fetch_latest_data(str(u))
                if not df_latest.empty:
                    actual_prb = df_latest.iloc[-1].get('sum_requested_prbs', None)
                    if actual_prb is not None:
                        self.actual_list.append(actual_prb)
                        self.predicted_list.append(pred_value)
                        logging.info("Due prediction: Predicted: %.4f, Actual: %.4f", pred_value, actual_prb)
                        print(f"Due prediction: Predicted: {pred_value:.4f}, Actual: {actual_prb:.4f}")

            # Fetch and predict
            df_latest = fetch_latest_data(str(u))
            if not df_latest.empty:
                
                scaled_data = preprocess_data(df_latest, scaler, FEATURES)
                if scaled_data is not None:
                    self.sequence = scaled_data
                    #self.sequence = self.sequence.reshape(1, SEQUENCE_LENGTH, len(FEATURES))
                    prediction = make_prediction(model, scaler, self.sequence, FEATURES)

                    if prediction is not None:
                        logging.info("Prediction: %.4f", prediction)
                        print(f"Prediction: {prediction:.4f}")
                        due_time = time.time() + PREDICTION_OFFSET
                        self.predictions_buffer.append((due_time, prediction))
            # Update plots
            
            # if self.actual_list and self.predicted_list:
            #     x_data = list(range(len(self.actual_list)))
            #     self.line_actual.set_data(x_data, self.actual_list)
            #     self.line_pred.set_data(x_data, self.predicted_list)
            #     self.ax1.axhline(y=50, color='black', linestyle='--')
            #     self.ax1.relim()
            #     self.ax1.autoscale_view()
            #     self.fig1.canvas.draw()
            #     self.fig1.canvas.flush_events()

            #     self.line_actual2.set_data(x_data, np.abs(np.array(self.actual_list) - 50))
            #     self.ax2.relim()
            #     self.ax2.autoscale_view()
            #     self.fig2.canvas.draw()
            #     self.fig2.canvas.flush_events()

            #     self.line_pred2.set_data(x_data, np.abs(np.array(self.predicted_list) - 50))
            #     self.ax3.relim()
            #     self.ax3.autoscale_view()
            #     self.fig3.canvas.draw()
            #     self.fig3.canvas.flush_events()
            
            return prediction


        except Exception as e:
            logging.error("Error in predict_and_visualize: %s", e)

    @xAppBase.start_function
    def start(self, e2_node_id, kpm_report_style, ue_ids, metric_names):
        """Start the xApp, subscribe to E2 node, and run prediction loop."""
        try:
            report_period = 1000
            granul_period = 100
            kpm_report_style = 2
            metric_names = ['DRB.UEThpDl', 'DRB.UEThpUl', 'RRU.PrbAvailDl', 'dl.cqi', 'ul.sinr', 'dl.mcs',
                            'dl.BufferBytes', 'tx.PktsDownlink', 'ul.mcs', 'ul.BufferBytes', 'rx.PktsUplink',
                            'rx.PktsUplinkErPer', 'tx.PktsUplinkErPer', 'phr']
            subscription_callback = lambda agent, sub, hdr, msg: self.my_subscription_callback(agent, sub, hdr, msg, kpm_report_style, ue_ids[0])
            subscription_callback_2 = lambda agent, sub, hdr, msg: self.my_subscription_callback(agent, sub, hdr, msg, kpm_report_style, ue_ids[1])
            print(f"Subscribe to E2 node ID: {e2_node_id}, RAN func: e2sm_kpm, Report Style: {kpm_report_style}, metrics: {metric_names}")
            #self.e2sm_kpm.subscribe_report_service_style_5(e2_node_id, report_period, ue_ids, metric_names, granul_period, subscription_callback)
            #self.e2sm_rc.control_slice_level_prb_quota(e2_node_id, ue_ids[0], 1, 70, dedicated_prb_ratio=100, ack_request=1)
            self.e2sm_kpm.subscribe_report_service_style_2(e2_node_id, report_period, ue_ids[0], metric_names, granul_period, subscription_callback)
            #self.e2sm_kpm.subscribe_report_service_style_2(e2_node_id, report_period, ue_ids[1], metric_names, granul_period, subscription_callback_2)



            


            

            # Start prediction loop
            logging.info("Starting prediction and visualization loop.")
            #while True:
            #self.predict_and_visualize()
            time.sleep(1)
        

        except KeyboardInterrupt:
            print("except 1")
            logging.info("Prediction loop terminated by user.")
            self.stop()

        except Exception as e:
            print("except 22")
            logging.error("Error in start method: %s", e)
            self.stop()

    # def signal_handler(self, signum, frame):
    #     """Handle termination signals and generate final plots."""
    #     logging.info("Terminating xApp due to signal %d", signum)
    #     if self.actual_list and self.predicted_list:
    #         mse = mean_squared_error(self.actual_list, self.predicted_list)
    #         mae = mean_absolute_error(self.actual_list, self.predicted_list)
    #         r2 = r2_score(self.actual_list, self.predicted_list)
    #         logging.info("Evaluation Metrics: MSE: %.4f, MAE: %.4f, R2: %.4f", mse, mae, r2)
    #         print("\nEvaluation Metrics:")
    #         print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    #         plt.figure(figsize=(12, 6))
    #         plt.plot(self.actual_list, marker='o', linestyle='-', label='Actual PRB')
    #         plt.plot(self.predicted_list, marker='x', linestyle='--', label='Predicted PRB')
    #         plt.title('Final: Actual vs. Predicted PRB')
    #         plt.xlabel('Measurement Index')
    #         plt.ylabel('PRB Value')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.savefig('final_prb_plot.png')

    #         plt.figure(figsize=(8, 8))
    #         plt.scatter(self.actual_list, self.predicted_list, color='blue', alpha=0.7, label='Data Points')
    #         min_val = min(min(self.actual_list), min(self.predicted_list))
    #         max_val = max(max(self.actual_list), max(self.predicted_list))
    #         plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    #         plt.xlabel('Actual PRB')
    #         plt.ylabel('Predicted PRB')
    #         plt.title('Final: Scatter Plot of Actual vs. Predicted PRB')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.savefig('final_scatter_plot.png')
    #     self.stop()

# ----------------------------- Main Block -----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KPM Predictor xApp')
    parser.add_argument("--config", type=str, default='', help="xApp config file path")
    parser.add_argument("--http_server_port", type=int, default=8092, help="HTTP server listen port")
    parser.add_argument("--rmr_port", type=int, default=4560, help="RMR port")
    parser.add_argument("--e2_node_id", type=str, default='gnbd_001_001_00019b_0', help="E2 Node ID")
    parser.add_argument("--ran_func_id", type=int, default=2, help="RAN function ID")
    parser.add_argument("--kpm_report_style", type=int, default=2, help="KPM report style")
    parser.add_argument("--ue_ids", type=str, default='0', help="UE ID")
    parser.add_argument("--metrics", type=str, default='DRB.UEThpDl', help="Metrics name as comma-separated string")

    args = parser.parse_args()
    config = args.config
    e2_node_id = args.e2_node_id
    ran_func_id = args.ran_func_id
    ue_ids = [0,1]
    kpm_report_style = args.kpm_report_style
    metrics = args.metrics.split(",")

    # Create and start xApp
    xapp = KpmPredictorXapp(config, args.http_server_port, args.rmr_port)
    xapp.e2sm_kpm.set_ran_func_id(ran_func_id)
    xapp.e2sm_rc.set_ran_func_id(3)

    signal.signal(signal.SIGQUIT, xapp.signal_handler)
    signal.signal(signal.SIGTERM, xapp.signal_handler)
    signal.signal(signal.SIGINT, xapp.signal_handler)
    xapp.start(e2_node_id, kpm_report_style, ue_ids, metrics)