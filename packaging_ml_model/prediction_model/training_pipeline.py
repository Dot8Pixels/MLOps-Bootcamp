import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.pipeline as pipe  # noqa: E402
from prediction_model.config import config  # noqa: E402
from prediction_model.processing.data_handling import (  # noqa: E402
    load_dataset,
    save_pipeline,
)


def perform_training() -> None:
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map({"N": 0, "Y": 1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES], train_y)
    save_pipeline(pipe.classification_pipeline)


if __name__ == "__main__":
    perform_training()
