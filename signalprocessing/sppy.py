# import pandas as pd
# import matplotlib.pyplot as plt

# # -------------------------------
# # Step 1 & 2: Load & extract event (as before)
# # -------------------------------
# file_path = "21418_20090709to20110713_qc.csv"
# df = pd.read_csv(file_path, sep=r"\s+", engine="python")
# df = df.iloc[:, :4]
# df.columns = ["Datetime", "Obs", "Tide", "Residual"]
# df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
# df.dropna(subset=["Datetime"], inplace=True)
# df.set_index("Datetime", inplace=True)
# df = df[["Residual"]]

# start = "2011-03-11 00:00:00"
# end   = "2011-03-12 00:00:00"
# df_event = df.loc[start:end]

# # -------------------------------
# # Step 3: Smooth with rolling window
# # -------------------------------
# window_size = 3  # 3 samples → 45 min smoothing
# df_event["Smoothed"] = df_event["Residual"].rolling(window=window_size, center=True).mean()

# # -------------------------------
# # Step 4: Plot raw vs smoothed
# # -------------------------------
# plt.figure(figsize=(14,6))
# plt.plot(df_event.index, df_event["Residual"], color="gray", linewidth=1, alpha=0.7, label="Raw")
# plt.plot(df_event.index, df_event["Smoothed"], color="blue", linewidth=2, label="Smoothed")
# plt.axvline(pd.Timestamp("2011-03-11 05:46:00"), color="red", linestyle="--", label="Earthquake Time")
# plt.title("DART Station 21418 - Raw vs Smoothed Residual Signal (March 11, 2011)")
# plt.xlabel("Time (UTC)")
# plt.ylabel("Residual Pressure (dbar)")
# plt.legend()
# plt.grid(True)
# plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

RAW_FILE = r"C:\Users\Anagha A Nagarhalli\Desktop\signalprocessing\21418_20090709to20110713_qc.csv"
FEATURE_FILE = r"C:\Users\Anagha A Nagarhalli\Desktop\signalprocessing\window_features.csv"

# 1. Load CSV using pandas, skip extra columns
df_raw = pd.read_csv(RAW_FILE, engine="python")
print("Raw columns:", df_raw.columns.tolist())

# 2. Identify columns
datetime_col = None
residual_col = None
for c in df_raw.columns:
    if "datetime" in c.lower() or "iso8601" in c.lower():
        datetime_col = c
    if "residual" in c.lower():
        residual_col = c

print("Datetime column:", datetime_col)
print("Residual column:", residual_col)

# 3. Extract only the needed columns
df = df_raw[[datetime_col, residual_col]].copy()
df.columns = ["Datetime", "Residual"]

# 4. Convert
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df["Residual"] = pd.to_numeric(df["Residual"], errors="coerce")

df.dropna(inplace=True)
df.set_index("Datetime", inplace=True)
print("Dataset shape after cleaning:", df.shape)
print(df.head())

# 5. Extract event window
start = "2011-03-11 00:00:00"
end = "2011-03-12 00:00:00"
df_event = df.loc[start:end].copy()
print("Event window shape:", df_event.shape)

# 6. Smooth signal
df_event["Smoothed"] = df_event["Residual"].rolling(window=3, center=True).mean()

# 7. Windowed feature extraction
window_size = 15
step_size = 5
features_list = []

for start_idx in range(0, len(df_event) - window_size + 1, step_size):
    window = df_event["Smoothed"].iloc[start_idx:start_idx + window_size].dropna().values
    if len(window) == 0:
        continue
    features_list.append({
        "start_time": df_event.index[start_idx],
        "end_time": df_event.index[start_idx + window_size - 1],
        "mean": np.mean(window),
        "std": np.std(window),
        "rms": np.sqrt(np.mean(window**2)),
        "skewness": skew(window),
        "kurtosis": kurtosis(window),
        "peak_amplitude": np.max(np.abs(window)),
        "zero_crossings": np.sum(np.diff(np.sign(window)) != 0)
    })

df_windows = pd.DataFrame(features_list)
print("Windowed features shape:", df_windows.shape)

# 8. Event detection
if not df_windows.empty:
    threshold = df_windows["peak_amplitude"].mean() + 2*df_windows["peak_amplitude"].std()
    df_windows["Event_Flag"] = (df_windows["peak_amplitude"] > threshold).astype(int)
    print("Threshold:", threshold)
    print("Flagged windows:", df_windows["Event_Flag"].sum())
else:
    print("No windows to flag")
    df_windows["Event_Flag"] = []

df_windows.to_csv(FEATURE_FILE, index=False)
print("Windowed features saved to:", FEATURE_FILE)

# 9. Plot
plt.figure(figsize=(14,6))
plt.plot(df_event.index, df_event["Residual"], color="gray", alpha=0.6, label="Raw")
plt.plot(df_event.index, df_event["Smoothed"], color="blue", linewidth=2, label="Smoothed Residual")

if not df_windows.empty:
    for idx, row in df_windows.iterrows():
        if row["Event_Flag"] == 1:
            plt.axvspan(row["start_time"], row["end_time"], color="orange", alpha=0.3)

plt.axvline(pd.Timestamp("2011-03-11 05:46:00"), color="red", linestyle="--", label="Event Time")
plt.title("DART Station 21418 - Smoothed Residual with Event Flags")
plt.xlabel("Time (UTC)")
plt.ylabel("Residual Pressure (dbar)")
plt.legend()
plt.grid(True)
plt.show()
