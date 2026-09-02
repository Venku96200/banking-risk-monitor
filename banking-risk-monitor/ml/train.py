"""
Train the Isolation Forest using 100,000 transactions
from bank_fraud.csv.

The model is unsupervised:
is_fraud and fraud_type are never used.
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Dataset
# --------------------------------------------------

DATA_PATH = "data/bank_fraud.csv"

MODEL_PATH = "ml/model.pkl"

SAMPLE_SIZE = 100_000


# --------------------------------------------------
# 2. Create the same 9 features used by the API
# --------------------------------------------------

def create_features(df):

    # Make timestamp
    df["timestamp"] = pd.to_datetime(
        df["transaction_date"] + " " + df["transaction_time"]
    )

    # Sort chronologically for every customer
    df = df.sort_values(
        ["customer_id", "timestamp"]
    ).reset_index(drop=True)


    # ----------------------------------------------
    # Feature 1 & 2: Amount
    # ----------------------------------------------

    customer_average = (
        df.groupby("customer_id")["transaction_amount"]
        .transform("mean")
    )

    amount_log = np.log1p(
        df["transaction_amount"]
    )

    amount_ratio = (
        df["transaction_amount"]
        / customer_average.clip(lower=1)
    )


    # ----------------------------------------------
    # Feature 3: Transactions in previous 10 min
    # ----------------------------------------------

    velocity_10m = []

    # ----------------------------------------------
    # Feature 4: Transactions in previous 24 hours
    # ----------------------------------------------

    velocity_24h = []

    # ----------------------------------------------
    # Feature 5: Minutes since previous transaction
    # ----------------------------------------------

    minutes_since_previous = []


    # Process each customer separately
    for customer_id, group in df.groupby("customer_id"):

        times = group["timestamp"].astype(
            "int64"
        ).to_numpy()

        # Convert nanoseconds → minutes
        times_minutes = times / (60 * 1_000_000_000)

        for i in range(len(group)):

            current_time = times_minutes[i]

            # Previous transactions only
            previous_times = times_minutes[:i]


            # 10-minute velocity
            count_10m = np.sum(
                previous_times >= current_time - 10
            )

            velocity_10m.append(count_10m)


            # 24-hour velocity
            count_24h = np.sum(
                previous_times >= current_time - 1440
            )

            velocity_24h.append(count_24h)


            # Time since previous transaction
            if i == 0:
                minutes_since_previous.append(1440)
            else:
                minutes_since_previous.append(
                    current_time - previous_times[-1]
                )


    # ----------------------------------------------
    # Feature 6: New device
    # ----------------------------------------------

    new_device = (
        ~df.duplicated(
            ["customer_id", "device_type"],
            keep="first"
        )
    ).astype(int)


    # ----------------------------------------------
    # Feature 7: New location
    # ----------------------------------------------

    new_location = (
        ~df.duplicated(
            ["customer_id", "city"],
            keep="first"
        )
    ).astype(int)


    # ----------------------------------------------
    # Feature 8: Transaction hour
    # ----------------------------------------------

    hour = df["timestamp"].dt.hour


    # ----------------------------------------------
    # Feature 9: Account age
    # ----------------------------------------------

    account_age = (
        df["account_age_years"] * 365
    ).clip(lower=0)


    # ----------------------------------------------
    # Combine all 9 features
    # ----------------------------------------------

    features = np.column_stack([

        amount_log,

        amount_ratio,

        velocity_10m,

        velocity_24h,

        minutes_since_previous,

        new_device,

        new_location,

        hour,

        account_age
    ])


    return features


# --------------------------------------------------
# 3. Train model
# --------------------------------------------------

def train():

    print("Loading bank fraud dataset...")

    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded {len(df):,} transactions."
    )


    # ----------------------------------------------
    # IMPORTANT:
    # Create features BEFORE sampling.
    # ----------------------------------------------

    features = create_features(df)


    print(
        f"Created {features.shape[1]} features."
    )


    # ----------------------------------------------
    # Take 100,000 transactions
    # ----------------------------------------------

    if len(features) > SAMPLE_SIZE:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(features),
            SAMPLE_SIZE,
            replace=False
        )

        features = features[indices]


    print(
        f"Training on {len(features):,} transactions."
    )


    # ----------------------------------------------
    # StandardScaler + Isolation Forest
    # ----------------------------------------------

    model = make_pipeline(

        StandardScaler(),

        IsolationForest(
            n_estimators=300,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
    )


    # Train
    model.fit(features)


    # ----------------------------------------------
    # Save model
    # ----------------------------------------------

    with open(MODEL_PATH, "wb") as file:

        pickle.dump(
            model,
            file
        )


    print(
        f"Model saved to {MODEL_PATH}"
    )


# --------------------------------------------------
# 4. Run
# --------------------------------------------------

if __name__ == "__main__":

    train()