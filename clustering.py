import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import numpy as np


def run_clustering(customers_path, visits_path, n_clusters=3):
    customers = pd.read_csv(customers_path)
    visits = pd.read_csv(visits_path)
    visits["Kirish_vaqti"] = pd.to_datetime(visits["Kirish_vaqti"])
    visits["Chiqish_vaqti"] = pd.to_datetime(visits["Chiqish_vaqti"])
    visits["Duration_min"] = (visits["Chiqish_vaqti"] - visits["Kirish_vaqti"]).dt.total_seconds() / 60

    avg_duration = visits.groupby("Pasport_raqami")["Duration_min"].mean().reset_index()
    avg_duration.columns = ["Pasport_raqami", "AvgDuration"]

    visit_count = visits.groupby("Pasport_raqami").size().reset_index(name="NumVisits")

    df = pd.merge(customers, avg_duration, on="Pasport_raqami", how="left")
    df = pd.merge(df, visit_count, on="Pasport_raqami", how="left")

    df["Jins_encoded"] = LabelEncoder().fit_transform(df["Jins"].fillna(""))
    df["Credit_encoded"] = LabelEncoder().fit_transform(df["Credit_Class"].fillna("C"))
    df["AvgDuration"].fillna(0, inplace=True)
    df["NumVisits"].fillna(0, inplace=True)

    features = df[["Yosh", "Jins_encoded", "Credit_encoded", "AvgDuration", "NumVisits"]]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(features_scaled)

    labels = []
    for i in range(n_clusters):
        segment = df[df["Cluster"] == i]
        age = segment["Yosh"].mean()
        visits = segment["NumVisits"].mean()
        credit = segment["Credit_Class"].mode()[0]

        if visits >= 8:
            label = "💎 VIP Mijozlar"
        elif credit in ["A", "B"]:
            label = "📈 Sodiq va Yaxshi Reytingli"
        elif credit in ["D", "E"]:
            label = "⚠️ Potensial Riskli Mijozlar"
        else:
            label = "👤 Yangi yoki Aralash Guruh"
        labels.append(label)

    df["Cluster_Label"] = df["Cluster"].apply(lambda x: labels[x])

    return df[["Pasport_raqami", "Ism", "Yosh", "Jins", "Credit_Class", "AvgDuration", "NumVisits", "Cluster", "Cluster_Label"]]
