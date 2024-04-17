import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))
import prediction_model.processing.preprocessing as pp  # noqa: E402
from config import config  # noqa: E402

classification_pipeline = Pipeline(
    [
        (
            "DomainProcessing",
            pp.DomainProcessing(
                variable_to_modify=config.FEATURE_TO_MODIFY,
                variable_to_add=config.FEATURE_TO_ADD,
            ),
        ),
        ("MeanImputation", pp.MeanImputer(variables=config.NUM_FEATURES)),
        ("ModeImputation", pp.ModeImputer(variables=config.CAT_FEATURES)),
        ("DropFeatures", pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ("LabelEncoder", pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ("LogTransform", pp.LogTransforms(variables=config.LOG_FEATURES)),
        ("MinMaxScale", MinMaxScaler()),
        ("LogisticClassifier", LogisticRegression(random_state=0)),
    ]
)
