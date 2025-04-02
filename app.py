
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tempfile
import os
import time

st.set_page_config(layout="wide")
st.title("🌊 Water Level Multi-Task Forecasting")

# Upload inputs
st.sidebar.header("🔧 Configuration")
model_file = st.sidebar.file_uploader("STEP 1: Upload trained LSTM model (.h5)", type=["h5"])
input_file = st.sidebar.file_uploader("STEP 2: Upload dataset CSV (datetime in 1st column)", type=["csv"])
fps = st.sidebar.slider("Animation Speed (Frames per Second)", 1, 10, 4)

if model_file and input_file:
    if st.button("🚀 STEP 3: Start Forecasting"):
        try:
            forecast_start = time.time()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(model_file.getbuffer())
                tmp_path = tmp.name
            model = load_model(tmp_path,compile=False)

            # Load and scale data
            df_raw = pd.read_csv(input_file)
            time_index = pd.to_datetime(df_raw.iloc[:, 0])
            df = df_raw.iloc[:, 1:]
            scaler = MinMaxScaler()
            scaler.fit(df.iloc[:, :])
            data_scaled = scaler.transform(df)

            # Parameters
            Lag_days = 24
            k_tasks = 36

            def series_to_supervised(data, n_in=1, n_out=1):
                df = pd.DataFrame(data)
                cols = []
                for i in range(n_in, 0, -1):
                    cols.append(df.shift(i))
                for i in range(n_out):
                    cols.append(df.shift(-i))
                agg = pd.concat(cols, axis=1)
                agg.dropna(inplace=True)
                return agg.values

            st.info("⏳ Preparing test data...")
            data_sequence = series_to_supervised(data_scaled, Lag_days, k_tasks)
            data_test = data_sequence[-5000:]

            test_x1_pr = data_test[:, 0::6].reshape(-1, Lag_days + k_tasks, 1)
            test_x2_pr = data_test[:, 1::6].reshape(-1, Lag_days + k_tasks, 1)
            test_x3_pr = data_test[:, 2::6].reshape(-1, Lag_days + k_tasks, 1)
            test_x4_pr = data_test[:, 3::6].reshape(-1, Lag_days + k_tasks, 1)
            test_x5_pr = data_test[:, 4::6].reshape(-1, Lag_days + k_tasks, 1)
            test_inflow = data_test[:, 5::6].reshape(-1, Lag_days + k_tasks, 1)
            test_x_inflow = test_inflow[:, :Lag_days, :]
            test_y = test_inflow[:, Lag_days:, :]

            test_x = [test_x_inflow, test_x1_pr, test_x2_pr, test_x3_pr, test_x4_pr, test_x5_pr]

            st.info("🔍 Forecasting with LSTM model...")
            y_pred_scaled = model.predict(test_x)
            q_max = np.max(df.iloc[:, 5])
            q_min = np.min(df.iloc[:, 5])
            y_pred = y_pred_scaled * (q_max - q_min) + q_min
            y_true = test_y * (q_max - q_min) + q_min
            y_true = y_true.reshape(y_true.shape[0], -1)
            y_pred = y_pred.reshape(y_pred.shape[0], -1)

            forecast_end = time.time()
            st.success(f"✅ Forecasting completed in {forecast_end - forecast_start:.2f} seconds.")

            gif_start = time.time()
            st.info("🎬 Generating animation...")

            full_obs = np.full((y_pred.shape[0] + k_tasks - 1,), np.nan)
            for i in range(y_true.shape[0]):
                full_obs[i:i + k_tasks] = np.where(np.isnan(full_obs[i:i + k_tasks]), y_true[i], full_obs[i:i + k_tasks])
            time_obs = time_index[-len(full_obs):].reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
            line_obs, = ax.plot([], [], 'b-', label='Observed', linewidth=2)
            line_pred, = ax.plot([], [], 'r--', label='Forecasted', linewidth=2)

            def init():
                ax.set_xlim(0, len(full_obs))
                ax.set_ylim(min(np.nanmin(full_obs), y_pred.min()), max(np.nanmax(full_obs), y_pred.max()) + 0.5)
                ax.set_xlabel("Datetime", fontsize=12)
                ax.set_ylabel("Water Level [m]", fontsize=12)

                tick_indices = np.linspace(0, len(time_obs) - 1, 8).astype(int)
                tick_labels = [time_obs[i].strftime("%b-%d\n%H:%M") for i in tick_indices]
                ax.set_xticks(tick_indices)
                ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=10)

                ax.grid(True)
                ax.legend(loc="upper right")
                plt.tight_layout()
                return line_obs, line_pred

            def update(frame):
                x_obs = np.arange(len(full_obs))
                x_pred = np.arange(frame, frame + k_tasks)
                y_pred_line = y_pred[frame]
                line_obs.set_data(x_obs, full_obs)
                line_pred.set_data(x_pred, y_pred_line)
                ax.set_title(f"📈 Forecast starting at {time_obs[frame].strftime('%Y-%m-%d %H:%M')}", fontsize=12, pad=0)
                return line_obs, line_pred

            ani = animation.FuncAnimation(fig, update, frames=len(y_pred), init_func=init, blit=True, repeat=True)

            gif_path = os.path.join(tempfile.gettempdir(), "Forecasting_movie.gif")
            ani.save(gif_path, writer='pillow', fps=fps)

            gif_end = time.time()
            st.success(f"✅ GIF created in {gif_end - gif_start:.2f} seconds.")

            with open(gif_path, "rb") as f:
                gif_bytes = f.read()
            st.image(gif_bytes, caption="📹 Forecasting Animation", use_container_width=True)
            st.download_button("Download GIF", gif_bytes, file_name="Forecasting_movie.gif", mime="image/gif")
        except Exception as e:
            st.error("❌ Something went wrong during forecasting. See error details below:")
            st.exception(e)
