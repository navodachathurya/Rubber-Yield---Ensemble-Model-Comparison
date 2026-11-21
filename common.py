import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}

# The only inputs a user can give are Year + Month, so these are the model features.
FEATURES = ["Year", "Month_num", "month_sin", "month_cos"]
TARGET = "Total (MT)"
START_YEAR = 2000  # first year in the dataset


def time_index(year, month_num):
    """Months since January 2000 (0, 1, 2, ...). ARIMA needs this to know 'how far ahead'."""
    return (np.asarray(year) - START_YEAR) * 12 + (np.asarray(month_num) - 1)


def build_features(year, month_num):
    """Turn Year + Month number into model features (works for one value or many)."""
    year = np.atleast_1d(year)
    month_num = np.atleast_1d(month_num)
    return pd.DataFrame({
        "Year": year,
        "Month_num": month_num,
        # sin/cos let the model know December is next to January (seasonality)
        "month_sin": np.sin(2 * np.pi * month_num / 12),
        "month_cos": np.cos(2 * np.pi * month_num / 12),
    })


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


class RFARIMAHybrid(BaseEstimator, RegressorMixin):

    def __init__(self, order=(1, 1, 1), n_estimators=200, random_state=42):
        self.order = order
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        t = time_index(X["Year"], X["Month_num"])
        order_idx = np.argsort(t)
        y_sorted = np.asarray(y, dtype=float)[order_idx]
        self.last_t_ = int(t.max())
        self.first_t_ = int(t.min())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.arima_ = ARIMA(y_sorted, order=self.order, trend="t" if self.order[1] == 1 else "c").fit()

        self.arima_in_sample_ = np.asarray(self.arima_.fittedvalues)
        # ARIMA with differencing has a meaningless first fitted value -> use the real value there
        self.arima_in_sample_[0] = y_sorted[0]

        residual = y_sorted - self.arima_in_sample_
        self.rf_ = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
        self.rf_.fit(X.iloc[order_idx], residual)
        return self

    def _arima_part(self, t):
        t = np.asarray(t)
        out = np.zeros(len(t), dtype=float)
        inside = t <= self.last_t_
        out[inside] = self.arima_in_sample_[t[inside] - self.first_t_]
        if (~inside).any():
            steps = int(t[~inside].max() - self.last_t_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = np.asarray(self.arima_.forecast(steps=steps))
            out[~inside] = fc[t[~inside] - self.last_t_ - 1]
        return out

    def predict(self, X):
        t = time_index(X["Year"], X["Month_num"]).astype(int)
        return self._arima_part(t) + self.rf_.predict(X)


class RFANNHybrid(BaseEstimator, RegressorMixin):
    """
    Hybrid for TABULAR regression:  ANN learns the smooth overall pattern, Random Forest fixes the residuals.
        prediction = ANN(main pattern) + RandomForest(residual part)
    """

    def __init__(self, hidden_layer_sizes=(32, 16), n_estimators=200, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        ann = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=3000,
                           random_state=self.random_state)
        self.ann_ = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), ann), transformer=StandardScaler())
        self.ann_.fit(X, y)
        residual = y - self.ann_.predict(X)
        self.rf_ = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
        self.rf_.fit(X, residual)
        return self

    def predict(self, X):
        return self.ann_.predict(X) + self.rf_.predict(X)
