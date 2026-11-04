import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def make_synthetic_housing(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the California Housing schema. For tests only."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(32.5, 41.9, n)
    lon = rng.uniform(-124.3, -114.3, n)
    med_inc = rng.lognormal(mean=1.2, sigma=0.5, size=n)
    house_age = rng.uniform(1, 52, n)
    ave_rooms = rng.uniform(2.0, 9.0, n)
    ave_bedrms = rng.uniform(0.8, 1.5, n)
    population = rng.uniform(100, 3000, n)
    ave_occup = rng.uniform(1.5, 5.0, n)
    coastal = np.exp(-((lon + 122.0) ** 2) / 8.0)
    target = 0.45 * med_inc + 1.2 * coastal + 0.02 * ave_rooms + rng.normal(0, 0.3, n)
    target = np.clip(target, 0.15, 5.0)
    return pd.DataFrame(
        {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": lat,
            "Longitude": lon,
            "MedHouseVal": target,
        }
    )


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_housing()
