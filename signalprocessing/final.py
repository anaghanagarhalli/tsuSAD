import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

# ========================================================
# 1. Load entire NOAA DART time series (raw residuals)
# ========================================================
RAW_FILE = r"C:\Users\Anagha A Nagarhalli\Desktop\signalprocessing\21418_20090709to20110713_qc.csv"
df = pd.read_csv(RAW_FILE, sep=r"\s+", engine="python")
df = df.iloc[:, :4]
df.columns = ["Datetime", "Obs", "Tide", "Residual"]
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
df.dropna(subset=["Datetime"], inplace=True)
df.set_index("Datetime", inplace=True)

# ========================================================
# 2. Extract tsunami event (March 11, 2011)
# ========================================================
event_start = "2011-03-11 00:00:00"
event_end = "2011-03-12 00:00:00"
df_event = df.loc[event_start:event_end].copy()

# ========================================================
# 3. Event processing: smoothing, baseline, filtering
# ========================================================
df_event["Smoothed"] = df_event["Residual"].rolling(window=3, win_type='gaussian', center=True).mean(std=1)
df_event["Smoothed"] = df_event["Smoothed"].bfill().ffill()
baseline_val = df_event["Smoothed"].iloc[50:150].mean()
df_event["Smoothed_BC"] = df_event["Smoothed"] - baseline_val

rho = 1025
g = 9.81
df_event["Wave_Height_m"] = df_event["Smoothed_BC"] / (rho * g)
h_deep, h_coast = 5645, 100
df_event["Wave_Height_Coast_m"] = df_event["Wave_Height_m"] * (h_deep / h_coast) ** 0.25

# ========================================================
# 4. Peak detection
# ========================================================
signal = df_event["Wave_Height_Coast_m"].dropna().values
time = df_event.dropna().index.values
threshold_factor = 1.0
peaks = []
median_val = np.median(signal)
mad = np.median(np.abs(signal - median_val))
while len(peaks) == 0 and threshold_factor > 0:
    adaptive_threshold = median_val + threshold_factor * mad
    found_peaks, props = find_peaks(signal, height=adaptive_threshold, distance=20)
    peaks = [p for p in found_peaks if props["peak_heights"][np.where(found_peaks == p)[0][0]] > 0.05]
    if len(peaks) == 0:
        threshold_factor -= 0.05

# Wave speeds for summary
c_deep = np.sqrt(g * h_deep)
c_coast = np.sqrt(g * h_coast)

# ========================================================
# 5. Analytical Plot 1: Full raw signal vs filtered event
# ========================================================
plt.figure(figsize=(18, 7))
plt.plot(df.index, df["Residual"], color="gray", alpha=0.5, label="Raw Residual (Full Data)")
plt.plot(df_event.index, df_event["Wave_Height_Coast_m"], color="blue", linewidth=2, label="Filtered Coastal Wave Height (Event Window)")
if len(peaks):
    plt.scatter(df_event.index[peaks], signal[peaks], color="red", s=60, label="Detected Peaks", zorder=10)
    for p in peaks:
        plt.annotate(f"{signal[p]:.2f} m", (df_event.index[p], signal[p]), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, color="darkred")
plt.title("Full Sensor Raw Residual vs. Filtered Coastal Wave Height (Event Window)")
plt.xlabel("Time (UTC)")
plt.ylabel("Amplitude (Raw: Residual, Filtered: Wave Height m)")
plt.legend()
plt.grid(True)
plt.xlim(df.index.min(), df.index.max())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ========================================================
# 6. Analytical Plot 2: Raw residual histogram (entire dataset)
# ========================================================
plt.figure(figsize=(11,5))
plt.hist(df["Residual"].values, bins=50, color="gray", alpha=0.7)
plt.title("Histogram of Raw Residuals (Entire Dataset)")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================================================
# 7. Analytical Plot 3: Moving average overlay on entire raw data
# ========================================================
window_ma = 10  # adjust for smoothing amount
moving_avg = df["Residual"].rolling(window=window_ma, center=True).mean()
plt.figure(figsize=(16,6))
plt.plot(df.index, df["Residual"], color="gray", alpha=0.5, label="Raw Residual (Full Data)")
plt.plot(df.index, moving_avg, color="firebrick", label=f"Moving Average (window={window_ma})")
plt.title(f"Raw Residual With Moving Average Overlay (window={window_ma})")
plt.xlabel("Time (UTC)")
plt.ylabel("Residual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================================================
# 8. Analytical Plot 4: Difference between raw & filtered (event window)
# ========================================================
plt.figure(figsize=(12,5))
plt.plot(df_event.index, df_event["Residual"] - df_event["Smoothed_BC"], color="purple")
plt.title("Difference: Raw Residual Minus Smoothed Baseline-Corrected (Event Window)")
plt.ylabel("Residual Difference")
plt.xlabel("Time (UTC)")
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
plt.show()

# ========================================================
# 9. Multi-panel event-focused plots (unchanged diagnostics)
# ========================================================
fig, axs = plt.subplots(3, 2, figsize=(18, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.3)
axs[0, 0].plot(df_event.index, df_event["Residual"], label="Raw Residual (Event Window)", color="gray")
axs[0, 0].set_title("Raw Residual (Event Window)")
axs[0, 0].set_ylabel("Residual")
axs[0, 0].legend()
axs[0, 0].grid(True)
axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axs[0, 1].plot(df_event.index, df_event["Wave_Height_Coast_m"], label="Coastal Wave Height (Processed)", color="blue")
if len(peaks):
    axs[0, 1].scatter(df_event.index[peaks], signal[peaks], color="red", label="Detected Peaks", zorder=5)
    for p in peaks:
        axs[0, 1].annotate(f"{signal[p]:.2f} m", (df_event.index[p], signal[p]),
            textcoords="offset points", xytext=(0,10), ha="center", fontsize=9, color="darkred")
axs[0, 1].set_title("Coastal Wave Height with Detected Peaks")
axs[0, 1].set_ylabel("Wave Height (m)")
axs[0, 1].legend()
axs[0, 1].grid(True)
axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axs[1, 0].plot(df_event.index, df_event["Smoothed_BC"], label="Smoothed & Baseline Corrected", color="green")
axs[1, 0].set_title("Smoothed & Baseline Corrected Residual")
axs[1, 0].set_ylabel("Residual")
axs[1, 0].legend()
axs[1, 0].grid(True)
axs[1, 1].hist(df_event["Wave_Height_Coast_m"].dropna(), bins=30, color="navy", alpha=0.7)
axs[1, 1].set_title("Histogram of Coastal Wave Heights")
axs[1, 1].set_xlabel("Wave Height (m)")
axs[1, 1].set_ylabel("Count")
if len(peaks):
    zoom_start = max(0, peaks[0] - 30)
    zoom_end = min(len(signal), peaks[0] + 60)
    axs[2, 0].plot(df_event.index[zoom_start:zoom_end], signal[zoom_start:zoom_end], label="Zoomed Coastal Wave Height", color="darkorange")
    axs[2, 0].scatter(df_event.index[peaks], signal[peaks], color="red", label="Detected Peaks", zorder=5)
    axs[2, 0].set_title("Zoomed Wave Height Around First Detected Peak")
    axs[2, 0].set_ylabel("Wave Height (m)")
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    axs[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
else:
    axs[2, 0].text(0.5, 0.5, "No Peaks Detected", ha='center', va='center', fontsize=12)
    axs[2, 0].set_title("Zoomed Wave Height Around First Detected Peak")
    axs[2, 0].axis('off')
axs[2, 1].axis('off')
for ax in axs.flatten():
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()

# ========================================================
# 10. Output summary table (unchanged)
# ========================================================
if len(peaks) == 0:
    first_peak_time, first_peak_amp = None, None
    second_peak_time, second_peak_amp = None, None
else:
    first_peak_time = df_event.index[peaks[0]]
    first_peak_amp = signal[peaks[0]]
    if len(peaks) > 1:
        second_peak_time = df_event.index[peaks[1]]
        second_peak_amp = signal[peaks[1]]
    else:
        second_peak_time, second_peak_amp = None, None

results = {
    "Parameter": [
        "First Peak Time",
        "First Peak Amplitude (m)",
        "Second Peak Time",
        "Second Peak Amplitude (m)",
        "Wave Speed Deep Ocean (m/s)",
        "Wave Speed Near Coast (m/s)"
    ],
    "Value": [
        first_peak_time,
        f"{first_peak_amp:.2f}" if first_peak_amp is not None else None,
        second_peak_time,
        f"{second_peak_amp:.2f}" if second_peak_amp is not None else None,
        f"{c_deep:.2f}",
        f"{c_coast:.2f}"
    ]
}
results_df = pd.DataFrame(results)
print(results_df)
