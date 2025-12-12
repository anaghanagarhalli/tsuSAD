import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# -------------------------------
# 1. File paths
# -------------------------------
RAW_FILE = r"C:\Users\Anagha A Nagarhalli\Desktop\signalprocessing\21418_20090709to20110713_qc.csv"
WINDOW_FEATURE_FILE = r"C:\Users\Anagha A Nagarhalli\Desktop\signalprocessing\window_features.csv"

# -------------------------------
# 2. Load CSV and preprocess
# -------------------------------
df = pd.read_csv(RAW_FILE, sep=r"\s+", engine="python")
df = df.iloc[:, :4]
df.columns = ["Datetime", "Obs", "Tide", "Residual"]
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
df.dropna(subset=["Datetime"], inplace=True)
df.set_index("Datetime", inplace=True)

# -------------------------------
# 3. Extract March 11, 2011 event
# -------------------------------
start = "2011-03-11 00:00:00"
end   = "2011-03-12 00:00:00"
df_event = df.loc[start:end].copy()
print("Extracted event shape:", df_event.shape)

# -------------------------------
# 4. Smooth signal
# -------------------------------
window_size = 3
df_event["Smoothed"] = df_event["Residual"].rolling(window=window_size, center=True).mean()

# -------------------------------
# 5. Compute windowed features
# -------------------------------
win_size = 15   # samples per window
step = 5        # step between windows
features_list = []

for start_idx in range(0, len(df_event) - win_size + 1, step):
    window = df_event["Smoothed"].iloc[start_idx:start_idx + win_size].dropna().values
    if len(window) == 0:
        continue
    features_list.append({
        "start_time": df_event.index[start_idx],
        "end_time": df_event.index[start_idx + win_size - 1],
        "mean": np.mean(window),
        "std": np.std(window),
        "rms": np.sqrt(np.mean(window**2)),
        "skewness": skew(window),
        "kurtosis": kurtosis(window),
        "peak_amplitude": np.max(np.abs(window)),
        "zero_crossings": np.sum(np.diff(np.sign(window)) != 0)
    })

df_windows = pd.DataFrame(features_list)

# -------------------------------
# 6. Flag high-amplitude windows
# -------------------------------
threshold = df_windows["peak_amplitude"].mean() + 2 * df_windows["peak_amplitude"].std()
df_windows["Event_Flag"] = (df_windows["peak_amplitude"] > threshold).astype(int)

# Save windowed features
df_windows.to_csv(WINDOW_FEATURE_FILE, index=False)
print(f"Windowed features saved to: {WINDOW_FEATURE_FILE}")
print(df_windows.head())

# -------------------------------
# 7. Plot smoothed signal with detected windows
# -------------------------------
plt.figure(figsize=(14,6))
plt.plot(df_event.index, df_event["Smoothed"], color="blue", label="Smoothed Signal")

# Highlight flagged windows
for idx, row in df_windows.iterrows():
    if row["Event_Flag"] == 1:
        plt.axvspan(row["start_time"], row["end_time"], color="orange", alpha=0.3)

# Mark earthquake time
plt.axvline(pd.Timestamp("2011-03-11 05:46:00"), color="red", linestyle="--", label="Earthquake Time")

plt.title("Smoothed Signal with Detected High-Amplitude Windows")
plt.xlabel("Time (UTC)")
plt.ylabel("Residual Pressure (dbar)")
plt.legend()
plt.grid(True)
plt.show()

