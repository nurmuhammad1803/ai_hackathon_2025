import os
import random
from datetime import timedelta
from faker import Faker
import pandas as pd
import numpy as np


def main():
    fake = Faker()
    random.seed(42)
    np.random.seed(42)

    base_dir = os.path.join(os.path.dirname(__file__), "data", "customer_analysis_data")
    os.makedirs(base_dir, exist_ok=True)

    n_customers = 500
    customers = []
    passport_ids = set()

    for _ in range(n_customers):
        while True:
            passport_id = fake.bothify(text='??######').upper()
            if passport_id not in passport_ids:
                passport_ids.add(passport_id)
                break

        customers.append({
            "Pasport_raqami": passport_id,
            "Ism": fake.first_name(),
            "Yosh": random.randint(18, 70),
            "Jins": random.choice(["Erkak", "Ayol"]),
            "Telefon": fake.phone_number(),
            "Credit_Class": random.choice(["A", "B", "C", "D", "E"])
        })

    df_customers = pd.DataFrame(customers)
    customers_csv = os.path.join(base_dir, "customers.csv")
    df_customers.to_csv(customers_csv, index=False)
    print(f"Saved {len(df_customers)} customers to {customers_csv}")

    n_visits = 1200
    visit_purposes = [
        "Yangi mashina ko‘rish", "Servis", "Test drive", "Sug'urta", "Texnik maslahat"
    ]
    passport_list = list(df_customers["Pasport_raqami"])
    visits = []

    for i in range(n_visits):
        passport_id = random.choice(passport_list)
        entry_time = fake.date_time_between(start_date='-30d', end_date='now')
        duration = timedelta(minutes=random.randint(5, 120))
        exit_time = entry_time + duration
        visits.append({
            "Tashrif_ID": i,
            "Pasport_raqami": passport_id,
            "Kirish_vaqti": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Chiqish_vaqti": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Maqsadi": random.choice(visit_purposes),
            "Tracked": random.choice(["Yes", "No"])
        })

    df_visits = pd.DataFrame(visits)
    visits_csv = os.path.join(base_dir, "visits.csv")
    df_visits.to_csv(visits_csv, index=False)
    print(f"Saved {len(df_visits)} visits to {visits_csv}")


if __name__ == "__main__":
    main()
