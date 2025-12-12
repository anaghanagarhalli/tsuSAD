import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load event
RAW_FILE = r"C:\Users\Anagha A Nagarhalli\Desktop\signalprocessing\21418_20090709to20110713_qc.csv"
df = pd.read_csv(RAW_FILE, sep=r"\s+", engine="python")
df = df.iloc[:, :4]
df.columns = ["Datetime", "Obs", "Tide", "Residual"]
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
df.dropna(subset=["Datetime"], inplace=True)
df.set_index("Datetime", inplace=True)

# Extract March 11, 2011
df_event = df.loc["2011-03-11 00:00:00":"2011-03-12 00:00:00"].copy()

# Smooth residual
df_event["Smoothed"] = df_event["Residual"].rolling(window=3, center=True).mean()

# Convert residual pressure (dbar) to deep-ocean wave height in meters
rho = 1025  # seawater density kg/m^3
g = 9.81    # gravity m/s^2
df_event["Wave_Height_Deep_m"] = df_event["Smoothed"] * 1e4 / (rho * g)

# -------------------------------
# 3. Estimate coastal wave height using Green's Law
# -------------------------------
h_deep = 4000   # deep ocean depth in m
h_coast = 20    # coastal depth in m
df_event["Wave_Height_Coast_m"] = df_event["Wave_Height_Deep_m"] * (h_deep/h_coast)**0.25

# -------------------------------
# 4. Estimate wave speed at coast
# -------------------------------
df_event["Wave_Speed_Coast_mps"] = np.sqrt(g * h_coast)  # shallow water approx

# -------------------------------
# 5. Plot results
# -------------------------------
plt.figure(figsize=(14,6))
plt.plot(df_event.index, df_event["Smoothed"], color="blue", label="Smoothed Residual (dbar)")
plt.plot(df_event.index, df_event["Wave_Height_Coast_m"], color="green", linestyle="--", label="Wave Height at Coast (m)")

# Highlight windows (example same as before, orange)
window_size = 15
step = 5
for start_idx in range(0, len(df_event) - window_size + 1, step):
    plt.axvspan(df_event.index[start_idx], df_event.index[start_idx + window_size - 1], color="orange", alpha=0.3)

plt.axvline(pd.Timestamp("2011-03-11 05:46:00"), color="red", linestyle="--", label="Earthquake Time")
plt.title("Tsunami Residual & Coastal Wave Height Estimate")
plt.xlabel("Time (UTC)")
plt.ylabel("Residual / Wave Height (m)")
plt.legend()
plt.grid(True)
plt.show()

# Print sample wave heights and speed
print("Sample coastal wave heights (m):")
print(df_event["Wave_Height_Coast_m"].head())
print("\nEstimated wave speed at coast (m/s):", df_event["Wave_Speed_Coast_mps"].iloc[0])
