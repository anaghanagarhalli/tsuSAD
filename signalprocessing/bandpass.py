import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load the CSV
# -------------------------------
file_path = "21418_20090709to20110713_qc.csv"

# Read CSV with whitespace as delimiter
df = pd.read_csv(file_path, sep=r"\s+", engine="python")

# Show structure
print("Number of columns:", len(df.columns))
print("Column names:", df.columns.tolist())

# -------------------------------
# Step 2: Keep useful columns
# -------------------------------
# Take only first 4 columns: datetime, obs, tide, residual
df = df.iloc[:, :4]
df.columns = ["Datetime", "Obs", "Tide", "Residual"]

# -------------------------------
# Step 3: Parse datetime
# -------------------------------
df["Datetime"] = pd.to_datetime(
    df["Datetime"], 
    format="%Y-%m-%dT%H:%M:%SZ",  # matches "2011-03-11T05:46:00Z"
    errors="coerce"
)

# Drop rows where datetime couldn’t be parsed
df.dropna(subset=["Datetime"], inplace=True)

# Set datetime as index
df.set_index("Datetime", inplace=True)

# -------------------------------
# Step 4: Keep only Residual
# -------------------------------
df = df[["Residual"]]

print("\nPreview of cleaned data:")
print(df.head())
print(df.info())

# -------------------------------
# Step 5: Plot full dataset
# -------------------------------
plt.figure(figsize=(14,6))
plt.plot(df.index, df["Residual"], color="blue", linewidth=0.8)
plt.title("DART Station 21418 - Residual Tsunami Signal (2009–2011)")
plt.xlabel("Time (UTC)")
plt.ylabel("Residual Pressure (dbar)")
plt.grid()
plt.show()

# -------------------------------
# Step 6: Debug datetime index
# -------------------------------
print("\nDatetime index sample:")
print(df.index[:5])
print("Index type:", type(df.index))

# -------------------------------
# Step 7: Extract March 11, 2011 event
# -------------------------------
# Adjust if timezone is included (+00:00)
start = "2011-03-11 00:00:00"
end   = "2011-03-12 00:00:00"

df_event = df.loc[start:end]

print("\nExtracted event shape:", df_event.shape)
print(df_event.head())
print(df_event.tail())

# -------------------------------
# Step 8: Plot March 11, 2011 event
# -------------------------------
plt.figure(figsize=(14,6))
plt.plot(df_event.index, df_event["Residual"], color="blue", linewidth=1)
plt.axvline(pd.Timestamp("2011-03-11 05:46:00"), color="red", linestyle="--", label="Earthquake Time (05:46 UTC)")
plt.title("DART Station 21418 - Residual Tsunami Signal (March 11, 2011)")
plt.xlabel("Time (UTC)")
plt.ylabel("Residual Pressure (dbar)")
plt.legend()
plt.grid()
plt.show()