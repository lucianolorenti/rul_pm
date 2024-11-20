"""Compute evaluating results of fitted models

One of the most important issues regarding PM is the ability to compare and evaluate different methods.

The main data structure used in the results module is a dictionary in which each of the keys is the model name, 
and the elements are a list of PredictionResult. 
Each element of the model array is interpreted as a Fold in CV settings. 

Additionally to the regression error, it is possible to compute some metrics more easily interpretable. In this context, two metrics were defined in [1], namely:

- Frequency of Unexpected Breaks (ρUB) - the percentage of failures not prevented;
- Amount of Unexploited Lifetime (ρUL) - the average number of time that could have been run before failure if the preventative maintenance suggested by the maintenance management mod-ule had not been performed.

When compaing between the predicted end of life with respect to the true end of that particular life three scenarios can happen:

- The predicted end of life occurs before the true one. In that case, the predictions were pessimistic and the tool could have been used more time.
- The remaining useful life arrives at zero after the true remaining useful life. In that case, we incur the risk of the tool breaking.  
- The predicted line coincides with the true line. In that case, we don’t have unexploited time, and the risk of breakage can be considered 0.

Since usually the breakages are considered more harmful, a possible approach to preventing unexpected failures is to consider a more conservative maintenance approach, providing maintenance tasks recommendations some time before the end of life predicted. In that way, a conservative window can be defined in which the recommendation of making maintenance task should be performed at time T-predicted - conservative window size.

[1] Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach
    Susto, G. A., Schirru, A., Pampuri, S., McLoone, S., & Beghi, A. (2015). 


"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from uncertainties import ufloat

from ceruleo.results.picewise_regression import (
    PiecewesieLinearFunction,
    PiecewiseLinearRegression,
)
from ceruleo.results.utils import compute_sample_weight, split_lives_indices

logger = logging.getLogger(__name__)



class MetricsResult(BaseModel):
    """An object that store regression metrics and times"""

    mae: float
    mse: float
    fitting_time: float =  Field(default=0)
    prediction_time: float = Field(default=0)


@dataclass
class PredictionResult:
    """A prediction result is composed by a name"""

    name: str
    true_RUL: np.ndarray
    predicted_RUL: np.ndarray
    metrics: MetricsResult = field(default_factory=lambda: MetricsResult(mae=0, mse=0))

    def compute_metrics(self):
        self.metrics.mae = mae(self.true_RUL, self.predicted_RUL)
        self.metrics.mse = mse(self.true_RUL, self.predicted_RUL)

    def __post_init__(self):    
        self.true_RUL = np.squeeze(self.true_RUL)
        self.predicted_RUL = np.squeeze(self.predicted_RUL)
        self.compute_metrics()



class FittedLife:
    """Represent a Fitted run-to-cycle failure

    Parameters:
        y_true: The true RUL target
        y_pred: The predicted target
        time: Time feature
        fit_line_not_increasing: Wether the fitted line can increase or not.
        RUL_threshold: Indicates the thresholding value used during  the fit

    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time: Optional[Union[np.ndarray, int]] = None,
        fit_line_not_increasing: bool = False,
        RUL_threshold: Optional[float] = None,
    ):
        self.fit_line_not_increasing = fit_line_not_increasing
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)

        if time is not None:
            self.degrading_start = FittedLife._degrading_start(y_true, RUL_threshold)
            if isinstance(time, np.ndarray):
                self.time = time
            else:
                self.time = np.linspace(start=0,stop=int(y_true[0]), num=len(y_true))

        else:
            self.degrading_start, self.time = FittedLife.compute_time_feature(
                y_true, RUL_threshold
            )

        # self.y_pred_fitted_picewise = self._fit_picewise_linear_regression(y_pred)
        # self.y_true_fitted_picewise = self._fit_picewise_linear_regression(y_true)

        self.RUL_threshold = RUL_threshold
        self.y_pred = y_pred
        self.y_true = y_true

        self.y_pred_fitted_coefficients = np.polyfit(self.time, self.y_pred, 1)
        p = np.poly1d(self.y_pred_fitted_coefficients)
        self.y_pred_fitted = p(self.time)

        self.y_true_fitted_coefficients = np.polyfit(self.time, self.y_true, 1)
        p = np.poly1d(self.y_true_fitted_coefficients)
        self.y_true_fitted = p(self.time)

    @staticmethod
    def compute_time_feature(
        y_true: np.ndarray, RUL_threshold: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """Compute the time feature based on the target

        Parameters:
            y_true: RUL target
            RUL_threshold:

        Returns:
            Degrading start time and time
        """
        degrading_start = FittedLife._degrading_start(y_true, RUL_threshold)
        time = FittedLife._compute_time(y_true, degrading_start)
        return degrading_start, time

    @staticmethod
    def _degrading_start(
        y_true: np.array, RUL_threshold: Optional[float] = None
    ) -> int:
        """
        Obtain the index when the life value is lower than the RUL_threshold

        Parameters:
            y_true: Array of true values of the RUL of the life
            RUL_threshold: float


        Return:
            If RUL_threshold is None, the degrading start if the first index.
            Otherwise it is the first index in which y_true < RUL_threshold
        """
        degrading_start = 0
        if RUL_threshold is not None:
            degrading_start_i = np.where(y_true < RUL_threshold)
            if len(degrading_start_i[0]) > 0:
                degrading_start = degrading_start_i[0][0]
        else:
            d = np.diff(y_true) == 0
            while (degrading_start < len(d)) and (d[degrading_start]):
                degrading_start += 1
        return degrading_start

    @staticmethod
    def _compute_time(y_true: np.ndarray, degrading_start_index: int) -> np.ndarray:
        """
        Compute the passage of time from the true RUL

        The passage of time is computed as the cumulative sum of the first
        difference of the true labels. In case there are tresholded values,
        the time steps of the thresholded zone is assumed to be as the median values
        of the time steps computed in the zones of the life in which we have information.

        Parameters:
            y_true: The true RUL labels
            degrading_start: The index in which the true RUL values starts to be lower than the treshold

        Returns:
            Time component
        """
        if len(y_true) == 1:
            return np.array([0])

        time_diff = np.diff(np.squeeze(y_true)[degrading_start_index:][::-1])
        time = np.zeros(len(y_true))
        if degrading_start_index > 0:
            if len(time_diff) > 0:
                time[0 : degrading_start_index + 1] = np.median(time_diff)
            else:
                time[0 : degrading_start_index + 1] = 1
        time[degrading_start_index + 1 :] = time_diff

        return np.cumsum(time)

    def _fit_picewise_linear_regression(self, y: np.ndarray) -> PiecewesieLinearFunction:
        """
        Fit the array trough a picewise linear regression

        Parameters:
            y: Points to be fitted
        Returns:
            The Picewise linear function fitted
        """
        pwlr = PiecewiseLinearRegression(not_increasing=self.fit_line_not_increasing)
        for j in range(len(y)):
            pwlr.add_point(self.time[j], y[j])
        line = pwlr.finish()

        return line

    def rmse(self, sample_weight=None) -> float:
        N = len(self.y_pred)
        sw = compute_sample_weight(sample_weight, self.y_true[:N], self.y_pred)
        return np.sqrt(np.mean(sw * (self.y_true[:N] - self.y_pred) ** 2))

    def mae(self, sample_weight=None) -> float:
        N = len(self.y_pred)
        sw = compute_sample_weight(sample_weight, self.y_true[:N], self.y_pred)
        return np.mean(sw * np.abs(self.y_true[:N] - self.y_pred))

    def noisiness(self) -> float:
        """
        How much the predictions resembles a line

        This metric is computed as the mse of the fitted values
        with respect to the least squares fitted line of this
        values

        Returns:
            The Mean Absolute Error of the fitted values with respect to the least squares fitted line
        """
        return mae(self.y_pred_fitted, self.y_pred)

    def slope_resemblance(self):
        m1 = self.y_true_fitted_coefficients[0]
        m2 = self.y_pred_fitted_coefficients[0]
        d = np.arctan((m1 - m2) / (1 + m1 * m2))
        d = d / (np.pi / 2)
        return 1 - np.abs((d / (np.pi / 2)))

    def predicted_end_of_life(self):
        z = np.where(self.y_pred == 0)[0]
        if len(z) == 0:
            return self.time[len(self.y_pred) - 1] + self.y_pred[-1]
        else:
            return self.time[z[0]]

    def end_of_life(self):
        z = np.where(self.y_true == 0)[0]
        if len(z) == 0:
            return self.time[len(self.y_pred) - 1] + self.y_true[-1]
        else:
            return self.time[z[0]]

    def maintenance_point(self, m: float = 0) -> float:
        """
        Compute the maintenance point

        The maintenance point is computed as the predicted end of life - m

        Parameters:
            m: Fault horizon  Defaults to 0.

        Returns:
            Time of maintenance
        """
        return self.predicted_end_of_life() - m

    def unexploited_lifetime(self, m: float = 0) -> float:
        """
        Compute the unexploited lifetime given a fault horizon window

        Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach
        Susto, G. A., Schirru, A., Pampuri, S., McLoone, S., & Beghi, A. (2015).

        Parameters:
            m: Fault horizon window. Defaults to 0.

        Returns:
            Unexploited lifetime
        """

        if self.maintenance_point(m) < self.end_of_life():
            return self.end_of_life() - self.maintenance_point(m)
        else:
            return 0

    def unexpected_break(self, m: float = 0, tolerance: float = 0) -> bool:
        """
        Compute weather an unexpected break will produce using a fault horizon window of size m

        Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach
        Susto, G. A., Schirru, A., Pampuri, S., McLoone, S., & Beghi, A. (2015).

        Parameters:
            m: Fault horizon window.

        Returns:
            A boolean indicating if an unexpected break will occur
        """
        if self.maintenance_point(m) - tolerance < self.end_of_life():
            return False
        else:
            return True





def split_lives(
    results: PredictionResult,
    RUL_threshold: Optional[float] = None,
    fit_line_not_increasing: bool = False,
    time: Optional[int] = None,
) -> List[FittedLife]:
    """
    Divide an array of predictions into a list of FittedLife Object

    Parameters:
        y_true: The true RUL target
        y_pred: The predicted RUL
        fit_line_not_increasing: Weather the fit line can increase, by default False
        time:  A vector with timestamps. If omitted will be computed from y_true

    Returns:
       FittedLife list
    """
    lives = []
    for r in split_lives_indices(results.true_RUL):
        if np.any(np.isnan(results.predicted_RUL[r])):
            continue
        lives.append(
            FittedLife(
                results.true_RUL[r],
                results.predicted_RUL[r],
                RUL_threshold=RUL_threshold,
                fit_line_not_increasing=fit_line_not_increasing,
                time=time,
            )
        )
    return lives

