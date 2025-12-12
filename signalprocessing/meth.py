# import pandas as pd
# import numpy as np

# # -------------------------------
# # Load CSV and preprocess
# # -------------------------------
# RAW_FILE = r"C:\Users\Anagha A Nagarhalli\Desktop\signalprocessing\21418_20090709to20110713_qc.csv"
# df = pd.read_csv(RAW_FILE, sep=r"\s+", engine="python")
# df = df.iloc[:, :4]
# df.columns = ["Datetime", "Obs", "Tide", "Residual"]
# df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
# df.dropna(subset=["Datetime"], inplace=True)
# df.set_index("Datetime", inplace=True)

# # -------------------------------
# # Extract March 11, 2011 event
# # -------------------------------
# df_event = df.loc["2011-03-11 00:00:00":"2011-03-12 00:00:00"].copy()
# df_event["Smoothed"] = df_event["Residual"].rolling(window=3, center=True).mean()

# # -------------------------------
# # Compute deep-ocean wave height (m)
# # -------------------------------
# rho = 1025  # seawater density kg/m^3
# g = 9.81    # gravity m/s^2
# df_event["Wave_Height_Deep_m"] = df_event["Smoothed"] * 1e4 / (rho * g)

# # -------------------------------
# # Estimate coastal wave height using Green's Law
# # -------------------------------
# h_deep = 4000   # deep ocean depth in m
# h_coast = 20    # coastal depth in m
# df_event["Wave_Height_Coast_m"] = df_event["Wave_Height_Deep_m"] * (h_deep/h_coast)**0.25

# # -------------------------------
# # Estimate wave speed at coast
# # -------------------------------
# df_event["Wave_Speed_Coast_mps"] = np.sqrt(g * h_coast)  # shallow water approx

# # -------------------------------
# # Display numerical output
# # -------------------------------
# print("Sample coastal wave heights (m):")
# print(df_event["Wave_Height_Coast_m"].head())
# print("\nEstimated wave speed at coast (m/s):", df_event["Wave_Speed_Coast_mps"].iloc[0])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. File paths
# -------------------------------
RAW_FILE = r"C:\Users\Anagha A Nagarhalli\Desktop\signalprocessing\21418_20090709to20110713_qc.csv"

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
# 4. Smooth signal (optional)
# -------------------------------
window_size = 3
df_event["Smoothed"] = df_event["Residual"].rolling(window=window_size, center=True).mean()

# -------------------------------
# 5. Convert residual pressure (dbar) to wave height (m)
# -------------------------------
rho = 1025  # kg/m^3
g = 9.81    # m/s^2

# Convert residual to meters
df_event["Wave_Height_m"] = df_event["Smoothed"] * 10000 / (rho * g)
print("Sample wave heights (m):")
print(df_event["Wave_Height_m"].head())

# -------------------------------
# 6. Estimate coastal wave height
# -------------------------------
h_deep = 5645  # meters (deep ocean)
h_coast = 100  # meters (nearshore)

df_event["Wave_Height_Coast_m"] = df_event["Wave_Height_m"] * (h_deep / h_coast)**0.25
print("\nSample coastal wave heights (m):")
print(df_event["Wave_Height_Coast_m"].head())

# -------------------------------
# 7. Calculate tsunami wave speed
# -------------------------------
c_deep = np.sqrt(g * h_deep)
c_coast = np.sqrt(g * h_coast)
print(f"\nEstimated wave speed in deep ocean (m/s): {c_deep:.2f}")
print(f"Estimated wave speed near coast (m/s): {c_coast:.2f}")

# -------------------------------
# 8. Plot deep ocean vs coastal wave height
# -------------------------------
plt.figure(figsize=(14,6))
plt.plot(df_event.index, df_event["Wave_Height_m"], label="Deep Ocean Wave Height (m)", color="blue")
plt.plot(df_event.index, df_event["Wave_Height_Coast_m"], label="Estimated Coastal Wave Height (m)", color="orange")
plt.xlabel("Time (UTC)")
plt.ylabel("Wave Height (m)")
plt.title("Tsunami Wave Heights: Deep Ocean vs Coastal Estimate")
plt.legend()
plt.grid(True)
plt.show()
