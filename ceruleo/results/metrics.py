

from typing import List
from ceruleo.results.results import FittedLife, PredictionResult, split_lives
import numpy as np 
from typing import Tuple

def unexploited_lifetime_from_cv(
    lives: List[List[FittedLife]], window_size: int, n: int
):
    std_per_window = []
    mean_per_window = []
    windows = np.linspace(0, window_size, n)
    for m in windows:
        jj = []
        for r in lives:
            ul_cv_list = [life.unexploited_lifetime(m) for life in r]

            jj.extend(ul_cv_list)
        mean_per_window.append(np.mean(jj))
        std_per_window.append(np.std(jj))

    return windows, np.array(mean_per_window), np.array(std_per_window)


def unexpected_breaks(
    d: List[PredictionResult], window_size: int, step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the risk of unexpected breaks with respect to the maintenance window size

    Parameters:
        d: Dictionary with the results
        window_size: Maximum size of the maintenance windows
        step: Number of points in which compute the risks.
            step different maintenance windows will be used.

    Returns:
        A tuple of np.arrays with:
            - Maintenance window size evaluated
            - Risk computed for every window size used
    """

    bb = [split_lives(fold) for fold in d]
    return unexpected_breaks_from_cv(bb, window_size, step)


def unexpected_breaks_from_cv(
    lives: List[List[FittedLife]], window_size: int, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the risk of unexpected breaks given a Cross-Validation results

    Parameters:
        lives: Cross validation results.
        window_size: Maximum size of the maintenance window
        n: Number of points to evaluate the risk of unexpected breaks


    Returns:
        A tuple of np.arrays with:
            - Maintenance window size evaluated
            - Risk computed for every window size used
    """
    std_per_window = []
    mean_per_window = []
    windows = np.linspace(0, window_size, n)
    for m in windows:
        jj = []
        for r in lives:
            ul_cv_list = [life.unexpected_break(m) for life in r]
            jj.extend(ul_cv_list)
        mean_per_window.append(np.mean(jj))
        std_per_window.append(np.std(jj))
    return windows, np.array(mean_per_window), np.array(std_per_window)


def metric_J_from_cv(lives: List[List[FittedLife]], window_size: int, n: int, q1, q2):
    J = []
    windows = np.linspace(0, window_size, n)
    for m in windows:
        J_of_m = []
        for r in lives:
            ub_cv_list = np.array([life.unexpected_break(m) for life in r])
            ub_cv_list = (ub_cv_list / (np.max(ub_cv_list) + 0.0000000001)) * q1
            ul_cv_list = np.array([life.unexploited_lifetime(m) for life in r])
            ul_cv_list = (ul_cv_list / (np.max(ul_cv_list) + 0.0000000001)) * q2
            values = ub_cv_list + ul_cv_list
            mean_J = np.mean(values)
            std_ul_cv = np.std(values)
            J_of_m.append(mean_J)
        J.append(np.mean(J_of_m))
    return windows, J


def metric_J(d, window_size: int, step: int):
    lives_cv = [split_lives(cv) for cv in d]
    return metric_J_from_cv(lives_cv, window_size, step)



def unexploited_lifetime(d: PredictionResult, window_size: int, step: int):
    bb = [split_lives(cv) for cv in d]
    return unexploited_lifetime_from_cv(bb, window_size, step)
