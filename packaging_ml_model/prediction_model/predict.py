import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config  # noqa: E402
from prediction_model.processing.data_handling import load_pipeline  # noqa: E402

classification_pipeline = load_pipeline(config.MODEL_NAME)


def generate_predictions(data_input) -> dict[str, NDArray[Any]]:
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred == 1, "Y", "N")
    result = {"prediction": output}
    return result


if __name__ == "__main__":
    generate_predictions()
