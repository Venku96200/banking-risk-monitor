# import os    :- Not necessary for now
from pathlib import Path
from functools import lru_cache
import numpy as np
import pickle

MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "model.pkl"
    



@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once per application process."""

    if not MODEL_PATH.exists():
        return None

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def score(features):
    trained_model = load_model()

    if trained_model is None:
        return 0.0

    decision = float(
        trained_model.decision_function(
            np.array([features])
        )[0]
    )

    return max(
        0.0,
        min(100.0, 50 - decision * 200)
    )