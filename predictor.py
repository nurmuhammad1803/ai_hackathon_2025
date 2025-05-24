import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

model = None
le_gender = LabelEncoder()
le_purpose = LabelEncoder()


def train_model(customers_path, visits_path):
    global model, le_gender, le_purpose

    customers = pd.read_csv(customers_path)
    visits = pd.read_csv(visits_path)
    visits["Kirish_vaqti"] = pd.to_datetime(visits["Kirish_vaqti"])
    visits["Chiqish_vaqti"] = pd.to_datetime(visits["Chiqish_vaqti"])
    visits["Duration_min"] = (visits["Chiqish_vaqti"] - visits["Kirish_vaqti"]).dt.total_seconds() / 60

    df = pd.merge(visits, customers, on="Pasport_raqami", how="left")

    df = df[["Yosh", "Jins", "Maqsadi", "Duration_min"]].dropna()
    df["Jins"] = le_gender.fit_transform(df["Jins"])
    df["Maqsadi"] = le_purpose.fit_transform(df["Maqsadi"])

    X = df[["Yosh", "Jins", "Maqsadi"]]
    y = df["Duration_min"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


def predict_duration(yosh, jins, maqsad):
    global model, le_gender, le_purpose

    if model is None:
        raise ValueError("Model is not trained. Please call train_model first.")

    jins_encoded = le_gender.transform([jins])[0]
    maqsad_encoded = le_purpose.transform([maqsad])[0]

    X_new = pd.DataFrame([[yosh, jins_encoded, maqsad_encoded]], columns=["Yosh", "Jins", "Maqsadi"])
    prediction = model.predict(X_new)[0]
    return round(prediction, 2)
