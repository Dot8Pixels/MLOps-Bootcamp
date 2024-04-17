import os
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))
from config import config  # noqa: E402


# Load the dataset
def load_dataset(file_name) -> pd.DataFrame:
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data


# Serialization
def save_pipeline(pipeline_to_save) -> None:
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")


# Deserialization
def load_pipeline(_pipeline_to_load) -> Any:
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded from {save_path}")
    return model_loaded
