
from typing import List, Union, Literal
from pydantic import BaseModel, Field
import numpy as np
from enum import Enum

class SampleWeightType(str, Enum):
    """Sample weight type"""

    relative = "relative"
    one = "one"


def compute_sample_weight(sample_weight:SampleWeightType,
                           y_true:np.ndarray,
                             y_pred:np.ndarray, c: float = 0.9) -> Union[np.ndarray, Literal[1]]:
    """Compute the sample weight for the regression metrics

    Args:
        sample_weight (SampleWeightType): Type of sample weight
        y_true (np.ndarray): RUL True values
        y_pred (np.ndarray): RUL Predicted values
        c (float, optional): . Defaults to 0.9.

    Returns:
        Union[np.ndarray, Literal[1]]: _description_
    """

    if sample_weight == "relative":
        return np.abs(y_true - y_pred) / (np.clip(y_true, c, np.inf))
    else:
        return 1
    

def split_lives_indices(y_true: np.ndarray) -> List[range]:
    """Obtain a list of indices for each life

    Args:
        y_true: True vector with the RUL

    Returns:
        List[range]: A list with the indices belonging to each life
    """
    assert len(y_true) >= 2
    lives_indices = (
        [0]
        + (np.where(np.diff(np.squeeze(y_true)) > 0)[0] + 1).tolist()
        + [len(y_true)]
    )
    indices = []
    for i in range(len(lives_indices) - 1):
        r = range(lives_indices[i], lives_indices[i + 1])
        if len(r) <= 1:
            continue
        indices.append(r)
    return indices