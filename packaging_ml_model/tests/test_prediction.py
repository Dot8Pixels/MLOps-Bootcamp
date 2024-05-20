import sys
from pathlib import Path
from typing import Any

import pytest
from numpy import dtype, ndarray

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config  # noqa: E402
from prediction_model.predict import generate_predictions  # noqa: E402
from prediction_model.processing.data_handling import load_dataset  # noqa: E402


@pytest.fixture
def single_prediction() -> dict[str, ndarray[Any, dtype[Any]]]:
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result


def test_single_pred_not_none(single_prediction) -> None:  # output is not none
    assert single_prediction is not None


def test_single_pred_str_type(single_prediction) -> None:  # data type is string
    assert isinstance(single_prediction.get("prediction")[0], str)


def test_single_pred_validate(single_prediction) -> None:  # check the output is Y
    assert single_prediction.get("prediction")[0] == "Y"
