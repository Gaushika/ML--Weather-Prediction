# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step1: Import the necessary packages using import statement.
Step2: Read the given csv file using read_csv() method and print the number of contents to be
displayed using df.head().
Step3: Import KMeans and use for loop to cluster the data.
Step4: Predict the cluster and plot data graphs.
Step5: Print the outputs and end the program

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Gaushika RR 
RegisterNumber:  212225040091
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
print("Data Head:\n", data.head())
print("\nData Info:")
data.info()
print("\nData Null Status:\n", data.isnull())
print("\nData Null Sum:\n", data.isnull().sum())

# Drop the 'bat' column as it has too many missing values to be useful
data.drop('bat', axis=1, inplace=True)
print("\nData Null Sum after dropping 'bat' column:\n", data.isnull().sum())

# Drop rows with any remaining NaN values for clean clustering input
data.dropna(inplace=True)
print("\nData Null Sum after dropping remaining NaNs:\n", data.isnull().sum())

# Select only numerical features for clustering
# Exclude 'time' (object) and 'wind_direction' (object)
numerical_cols = [
    'hum', 'tem', 'co2', 'illumination', 'pressure', 'pm2_5', 'pm10',
    'wind_direction_angle', 'wind_speed', 'wind_speed_level', 'tsr'
]
X_cluster = data[numerical_cols]
print("\nFeatures used for clustering (X_cluster.head()):\n", X_cluster.head())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_cluster) # Fit on selected numerical features
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# Assuming 4 clusters based on Elbow Method (a common choice for similar datasets)
# You might need to adjust this based on the elbow plot for this specific dataset
km = KMeans(n_clusters=4, init="k-means++", random_state=42, n_init=10)
km.fit(X_cluster) # Fit on selected numerical features
y_pred = km.predict(X_cluster)
print("\nCluster Predictions (y_pred):\n", y_pred)

data["cluster"] = y_pred

df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]

plt.figure(figsize=(12, 8))
# Update scatter plot to use relevant columns from the weather dataset
# Using 'tem' (temperature) and 'hum' (humidity) as examples for visualization
plt.scatter(df0["tem"], df0["hum"], c="black", label="Cluster 0")
plt.scatter(df1["tem"], df1["hum"], c="cyan", label="Cluster 1")
plt.scatter(df2["tem"], df2["hum"], c="yellow", label="Cluster 2")
plt.scatter(df3["tem"], df3["hum"], c="blue", label="Cluster 3")
# If you choose 5 clusters, make sure to add df4 scatter plot
# plt.scatter(df4["tem"], df4["hum"], c="green", label="Cluster 4")

plt.xlabel("Temperature (tem)")
plt.ylabel("Humidity (hum)")
plt.title("Weather Station Data Clusters (Temperature vs. Humidity)")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
<img width="786" height="545" alt="image" src="https://github.com/user-attachments/assets/217ebc99-70c7-48ea-b724-e97329d0c37d" />
<img width="506" height="398" alt="image" src="https://github.com/user-attachments/assets/1b9989be-b121-496a-87c9-9dc3ed7434b1" />
<img width="408" height="343" alt="image" src="https://github.com/user-attachments/assets/dde0c58b-33c1-4495-b655-e9871f8a9356" />
<img width="480" height="648" alt="image" src="https://github.com/user-attachments/assets/46c67fbf-b937-4480-99f4-217dc9c398b7" />
<img width="610" height="305" alt="image" src="https://github.com/user-attachments/assets/a1d303b7-7da0-4d20-ac0c-1550ba058396" />
<img width="1005" height="616" alt="image" src="https://github.com/user-attachments/assets/1d32c8d1-6d86-4651-b31e-e7ba6395439f" />
<img width="750" height="594" alt="image" src="https://github.com/user-attachments/assets/d90c4517-4880-4cb4-ba65-8b17120363f6" />
<img width="1063" height="686" alt="image" src="https://github.com/user-attachments/assets/22187083-a9fb-4998-aacf-76af5d344f6d" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and
verified using python programming.
