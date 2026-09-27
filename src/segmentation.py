import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------
# 📌 Load Data
# -------------------------------
df = pd.read_csv("data/customer_transactions.csv")

# Convert date column
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# -------------------------------
# 📌 RFM Calculation
# -------------------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "count",
    "Amount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# -------------------------------
# 📌 Scaling
# -------------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# -------------------------------
# 📌 KMeans Clustering
# -------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# -------------------------------
# 📌 Cluster Labels (Business Meaning)
# -------------------------------
cluster_map = {
    0: "Low Value",
    1: "Mid Value",
    2: "High Value",
    3: "At Risk"
}

rfm["Segment"] = rfm["Cluster"].map(cluster_map)

print("\nRFM Table with Segments:\n")
print(rfm)

# -------------------------------
# 📊 VISUALIZATION 1: Scatter Plot
# -------------------------------
plt.figure()

segments = rfm["Segment"].unique()

for segment in segments:
    subset = rfm[rfm["Segment"] == segment]
    plt.scatter(
        subset["Frequency"],
        subset["Monetary"],
        label=segment
    )

plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Customer Segmentation (K-Means)")

plt.legend(title="Customer Segments")
plt.grid(True)

plt.show()
# -------------------------------
# 📊 VISUALIZATION 2: Cluster Distribution
# -------------------------------
plt.figure()

rfm["Cluster"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("Cluster Distribution")

plt.grid(True)
plt.show()

# -------------------------------
# 📊 VISUALIZATION 3: Avg Spending by Segment
# -------------------------------
plt.figure()

rfm.groupby("Segment")["Monetary"].mean().plot(kind="bar")

plt.title("Average Spending by Customer Segment")
plt.ylabel("Monetary")
plt.xticks(rotation=30)

plt.grid(True)
plt.show()