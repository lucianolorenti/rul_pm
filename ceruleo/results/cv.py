


from typing import List
from ceruleo.results.results import PredictionResult
import numpy as np

def cv_regression_metrics_single_model(
    results: List[PredictionResult], threshold: float = np.inf
):
    errors = {"MAE": [], "MAE SW": [], "MSE": [], "MSE SW": [], "MAPE": []}
    for result in results:
        y_mask = np.where(result.true_RUL <= threshold)[0]
        y_true = np.squeeze(result.true_RUL[y_mask])
        y_pred = np.squeeze(result.predicted_RUL[y_mask])
        mask = np.isfinite(y_pred)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        if len(np.unique(y_pred)) == 1:
            continue

        sw = compute_sample_weight(
            "relative",
            y_true,
            y_pred,
        )
        try:
            MAE_SW = mae(
                y_true,
                y_pred,
                sample_weight=sw,
            )
        except:
            MAE_SW = np.nan
        try:
            MAE = mae(y_true, y_pred)
        except:
            MAE = np.nan

        try:
            MSE_SW = mse(
                y_true,
                y_pred,
                sample_weight=sw,
            )
        except:
            MSE_SW = np.nan

        try:
            MSE = mse(y_true, y_pred)
        except:
            MSE = np.nan

        try:
            MAPE = mape(y_true, y_pred)
        except:
            MAPE = np.nan

        lives = split_lives(result)
        errors["MAE"].append(MAE)
        errors["MAE SW"].append(MAE_SW)
        errors["MSE"].append(MSE)
        errors["MSE SW"].append(MSE_SW)
        errors["MAPE"].append(MAPE)

    errors1 = {}
    for k in errors.keys():
        errors1[k] = ufloat(
            np.round(np.mean(errors[k]), 2), np.round(np.std(errors[k]), 2)
        )
    return errors1


def cv_regression_metrics(
    results_dict: Dict[str, List[PredictionResult]], threshold: float = np.inf
) -> dict:
    """
    Compute regression metrics for each model

    Parameters:
        data: Dictionary with the model predictions.
        threshold: Compute metrics errors only in RUL values less than the threshold

    Returns:
        A dictionary with the following structure:
            d: { ['Model]:
                    {
                        'MAE': {
                            'mean':
                            'std':
                        },
                        'MAE SW': {
                            'mean':
                            'std':
                        },
                        'MSE': {
                            'mean':
                            'std':
                        },
                    }
                ]

    """
    out = {}
    for model_name in results_dict.keys():
        out[model_name] = cv_regression_metrics_single_model(
            results_dict[model_name], threshold
        )
    return out




def model_cv_results(
    results: List[PredictionResult],
    nbins: Optional[int] = None,
    bin_edges: Optional[np.ndarray] = None,
) -> CVResults:
    if nbins is None and bin_edges is None:
        raise ValueError("nbins and bin_edges cannot be both None")
    if nbins is None:
        nbins = len(bin_edges) - 1
    if bin_edges is None:
        max_y_value = np.max([r.true_RUL.max() for r in results])
        bin_edges = np.linspace(0, max_y_value, nbins + 1)

    trues = []
    predicted = []
    for results in results:
        trues.append(results.true_RUL)
        predicted.append(results.predicted_RUL)
    return CVResults(trues, predicted, nbins=nbins, bin_edges=bin_edges)


def models_cv_results(
    results_dict: Dict[str, List[PredictionResult]], nbins: int
) -> Tuple[np.ndarray, Dict[str, CVResults]]:
    """Create a dictionary with the result of each cross validation of the model"""
    max_y_value = np.max(
        [
            r.true_RUL.max()
            for model_name in results_dict.keys()
            for r in results_dict[model_name]
        ]
    )
    bin_edges = np.linspace(0, max_y_value, nbins + 1)
    model_results = {}
    for model_name in results_dict.keys():
        model_results[model_name] = model_cv_results(
            results_dict[model_name], bin_edges=bin_edges
        )

    return bin_edges, model_results

class CVResults:
    """
    Compute the error histogram

    Compute the error with respect to the RUL considering the results of different
    folds

    Parameters:
        y_true: List with the true values of each hold-out set of a cross validation
        y_pred: List with the predictions of each hold-out set of a cross validation
        nbins: Number of bins to compute the histogram

    """

    def __init__(
        self,
        y_true: List[List],
        y_pred: List[List],
        nbins: int = 5,
        bin_edges: Optional[np.array] = None,
    ):
        if bin_edges is None:
            max_value = np.max([np.max(y) for y in y_true])
            bin_edges = np.linspace(0, max_value, nbins + 1)
        self.n_folds = len(y_true)
        assert bin_edges is not None
        self.n_bins = len(bin_edges) - 1
        self.bin_edges = bin_edges
        self.mean_error = np.zeros((self.n_folds, self.n_bins))
        self.mae = np.zeros((self.n_folds, self.n_bins))
        self.mse = np.zeros((self.n_folds, self.n_bins))
        self.errors = []
        for i, (y_pred, y_true) in enumerate(zip(y_pred, y_true)):
            self._add_fold_result(i, y_pred, y_true)

    def _add_fold_result(self, fold: int, y_pred: np.array, y_true: np.array):
        y_pred = np.squeeze(y_pred)
        y_true = np.squeeze(y_true)

        for j in range(len(self.bin_edges) - 1):
            mask = (y_true >= self.bin_edges[j]) & (y_true <= self.bin_edges[j + 1])
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            errors = y_true[indices] - y_pred[indices]

            self.mean_error[fold, j] = np.mean(errors)

            self.mae[fold, j] = np.mean(np.abs(errors))
            self.mse[fold, j] = np.mean((errors) ** 2)
            self.errors.append(errors)