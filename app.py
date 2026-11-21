"""
Streamlit dashboard - Rubber Yield: Ensemble Model Comparison (display only, no user input).

Run locally:   streamlit run app.py
It loads the models saved by rubber_ensemble_model.ipynb (./artifacts/*.pkl) directly,
so the FastAPI server (main.py) is NOT needed.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import FEATURES, MONTH_TO_NUM, TARGET, build_features, regression_metrics

BASE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE, "artifacts")

COLORS = {
    "random_forest": "#16a34a", "xgboost": "#ef4444", "voting": "#3b82f6",
    "stacking": "#8b5cf6", "hybrid_rf_arima": "#0d9488", "hybrid_rf_ann": "#db2777",
}
FORECAST_START_YEAR, FORECAST_MONTHS = 2024, 36

st.set_page_config(page_title="Rubber Yield - Model Comparison", page_icon="🌳", layout="wide")


# --------------------------------------------------------------------------- load + compare (runs once, then cached)
def _load(path):
    """Load a saved model (.pkl). Only load .pkl files that you created yourself."""
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading models...")
def load_all():
    meta = json.load(open(os.path.join(ART, "meta.json")))
    names = list(meta["models"].keys())
    label = {n: meta["models"][n]["name"] for n in names}
    group = {n: meta["models"][n]["group"] for n in names}

    # data (same preprocessing as the notebook)
    df = pd.read_excel(os.path.join(BASE, "data", "Dataset_of_Rubber.xlsx")).drop_duplicates()
    df["Month_num"] = df["Month"].map(MONTH_TO_NUM)
    df = df.sort_values(["Year", "Month_num"]).reset_index(drop=True)
    df["period"] = df["Year"].astype(str) + "-" + df["Month_num"].astype(str).str.zfill(2)

    eval_models = {n: _load(os.path.join(ART, "eval", f"{n}.pkl")) for n in names}
    final_models = {n: _load(os.path.join(ART, "final", f"{n}.pkl")) for n in names}

    # ---- test-set comparison: models trained on 2000-2020, tested on 2021-2023
    t0, t1 = meta["split"]["test"]
    test = df[(df["Year"] >= t0) & (df["Year"] <= t1)]
    Xt = build_features(test["Year"].values, test["Month_num"].values)
    test_preds = {n: eval_models[n].predict(Xt) for n in names}
    metrics = pd.DataFrame([{"model": n, "Model": label[n], "Group": group[n],
                             **regression_metrics(test[TARGET].values, test_preds[n])} for n in names])
    metrics = metrics.sort_values("RMSE").reset_index(drop=True)
    metrics.insert(0, "Rank", metrics.index + 1)

    # ---- forecast comparison after the last year of data
    idx = pd.period_range(start=f"{FORECAST_START_YEAR}-01", periods=FORECAST_MONTHS, freq="M")
    Xf = build_features(np.array([p.year for p in idx]), np.array([p.month for p in idx]))
    forecast = {n: final_models[n].predict(Xf) for n in names}
    f_periods = [f"{p.year}-{p.month:02d}" for p in idx]

    # ---- feature importance of tree models
    importance = pd.DataFrame({label[n]: final_models[n].feature_importances_
                               for n in ("random_forest", "xgboost")}, index=FEATURES)

    return dict(meta=meta, names=names, label=label, group=group, df=df, test=test, test_preds=test_preds,
                metrics=metrics, forecast=forecast, f_periods=f_periods, importance=importance)


if not os.path.exists(os.path.join(ART, "meta.json")):
    st.error("`artifacts/` not found. Run `rubber_ensemble_model.ipynb` first (Run All) to train and save the models.")
    st.stop()

_saved = set(json.load(open(os.path.join(ART, "meta.json")))["models"])
if _saved != set(COLORS):
    st.error(f"The saved models in `artifacts/` are out of date (found: {sorted(_saved - set(COLORS))}; "
             f"missing: {sorted(set(COLORS) - _saved)}). Delete the `artifacts/` folder, run "
             "`rubber_ensemble_model.ipynb` again (Run All) in this same folder, then restart the app.")
    st.stop()

D = load_all()
meta, label, metrics = D["meta"], D["label"], D["metrics"]
split = meta["split"]
best_test = metrics.iloc[0]


def style(fig, height=420, **kw):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), **kw)
    return fig


def line_fig(x, series, title, actual=None, dash=True, height=450):
    """series: {model_key: values}. Click a legend item to hide / show a line."""
    fig = go.Figure()
    if actual is not None:
        fig.add_scatter(x=x, y=actual, name="Actual", line=dict(color="#0f172a", width=4))
    for n, y in series.items():
        fig.add_scatter(x=x, y=y, name=label[n], line=dict(color=COLORS[n], width=2, dash="dash" if dash else "solid"))
    fig.update_yaxes(title="Total (MT)")
    return style(fig, height, title=title, hovermode="x unified")


# --------------------------------------------------------------------------- page
st.title("🌳 Rubber Yield - Ensemble Model Comparison")
st.caption("Which model predicts monthly rubber production (Total, MT) best? "
           "Display-only comparison for an ensemble learning assignment - not for real-world use.")

c1, c2, c3 = st.columns(3)
c1.metric("Best on test years", best_test["Model"], f"RMSE {best_test['RMSE']:.1f} MT", delta_color="off")
c2.metric("Models compared", len(D["names"]))
c3.metric("Data", f"{meta['data_years'][0]} - {meta['data_years'][1]}", f"{len(D['df'])} months", delta_color="off")

tab1, tab2, tab4, tab5 = st.tabs(
    ["📊 Model comparison", "📈 Actual vs predicted", "🔮 Forecast", "🗂️ Data & method"])

# ---- tab 1: test-set metrics
with tab1:
    st.subheader("Test-set comparison")
    st.caption(f"Trained on {split['train'][0]}-{split['validation'][1]}, tested on {split['test'][0]}-{split['test'][1]} "
               "(never seen during training or tuning). Lower error is better.")
    cols = st.columns(3)
    for col, (m, unit) in zip(cols, [("RMSE", "MT"), ("MAE", "MT"), ("MAPE", "%")]):
        d = metrics.sort_values(m, ascending=False)  # best bar at the top
        fig = go.Figure(go.Bar(x=d[m], y=d["Model"], orientation="h", marker_color=[COLORS[k] for k in d["model"]],
                               text=d[m].round(2 if m == "MAPE" else 1), textposition="outside"))
        fig.update_xaxes(title=unit, range=[0, d[m].max() * 1.2])
        col.plotly_chart(style(fig, 380, title=f"{m} ({unit})", showlegend=False))

    show = metrics[["Rank", "Model", "Group", "MAE", "RMSE", "MAPE", "R2"]].rename(columns={"MAPE": "MAPE %", "R2": "R²"})
    st.dataframe(show.round({"MAE": 1, "RMSE": 1, "MAPE %": 2, "R²": 2}), hide_index=True)
    st.info("**About R²:** the 2021-2023 values vary very little (≈10,025 - 10,165 MT), so R² is strongly negative for most "
            "models. Compare the models with MAE / RMSE / MAPE (error in MT or %).")

# ---- tab 2: actual vs predicted
with tab2:
    st.subheader("Actual vs predicted (test years)")
    st.caption("Click a model in the legend to hide / show it. Double-click to isolate one.")
    test = D["test"]
    st.plotly_chart(line_fig(test["period"], D["test_preds"], "Test years 2021-2023", actual=test[TARGET].values))

    err = pd.DataFrame({label[n]: np.abs(D["test_preds"][n] - test[TARGET].values) for n in D["names"]})
    err.insert(0, "Period", test["period"].values)
    fig = go.Figure()
    for n in D["names"]:
        fig.add_scatter(x=err["Period"], y=err[label[n]], name=label[n], line=dict(color=COLORS[n], width=2))
    fig.update_yaxes(title="Absolute error (MT)")
    st.plotly_chart(style(fig, 380, title="Absolute error per month", hovermode="x unified"))

# ---- tab 4: forecast
with tab4:
    st.subheader(f"Forecast comparison ({FORECAST_START_YEAR}-{FORECAST_START_YEAR + FORECAST_MONTHS // 12 - 1})")
    st.caption("How each model continues after the last year in the data (2023). Final models trained on all data. "
               "There is no actual value to compare with - the point is how differently the models behave.")
    hist = D["df"].tail(36)
    fig = line_fig(D["f_periods"], D["forecast"], "Forecast of every model", dash=False, height=480)
    fig.add_scatter(x=hist["period"], y=hist[TARGET], name="History", line=dict(color="#0f172a", width=4))
    st.plotly_chart(fig)

    f = pd.DataFrame(D["forecast"], index=D["f_periods"]).rename(columns=label)
    picks = [p for p in D["f_periods"] if p.endswith("-12")]
    st.markdown("**December values (MT)**")
    st.dataframe(f.loc[picks].round(1).rename_axis("Period").reset_index(), hide_index=True)
    st.warning("Tree-based models (Random Forest, XGBoost, Voting, Stacking) cannot extrapolate: after 2023 they "
               "repeat the level of the latest years, so their lines stay flat from year to year (Stacking sits at a different "
               "level but is flat too). Only the hybrid models carry a trend (from ARIMA or the ANN). "
               "Which is *right* cannot be known for future months.")

# ---- tab 5: data + method
with tab5:
    left, right = st.columns(2)
    with left:
        st.subheader("Dataset")
        y = D["df"][TARGET]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Min (MT)", f"{y.min():,.0f}")
        s2.metric("Max (MT)", f"{y.max():,.0f}")
        s3.metric("Mean (MT)", f"{y.mean():,.0f}")
        s4.metric("Std dev", f"{y.std():,.0f}")
        fig = go.Figure(go.Scatter(x=D["df"]["period"], y=y, line=dict(color="#16a34a", width=2), name="Total (MT)"))
        fig.update_yaxes(title="Total (MT)")
        st.plotly_chart(style(fig, 330, title="Monthly production 2000-2023", showlegend=False))
    with right:
        st.subheader("Feature importance")
        imp = D["importance"]
        fig = go.Figure()
        for col in imp.columns:
            fig.add_bar(x=imp.index, y=imp[col], name=col)
        fig.update_layout(barmode="group")
        st.plotly_chart(style(fig, 330, title="Tree-based models (final)"))
        st.caption("Almost all importance is on Year: the series is driven by long-term level shifts, not by the month.")

    st.subheader("Method")
    bp = meta["best_params"]
    st.markdown(f"""
- **Target:** `{meta['target']}`. **Inputs:** Year and Month only (plus sin/cos of the month).
- **Split by time:** train {split['train'][0]}-{split['train'][1]} · validation {split['validation'][0]}-{split['validation'][1]} · test {split['test'][0]}-{split['test'][1]}.
  Models are fitted on train + validation for the test comparison, and on all data for forecasting.
- **Models:** Random Forest, XGBoost · Voting (RF+XGB) · Stacking (RF+XGB → Ridge) ·
  Hybrid RF + ARIMA (ARIMA trend + RF on residuals) · Hybrid RF + ANN (ANN pattern + RF on residuals).
- **Tuned settings:** RF `{bp['random_forest']}` · XGBoost trees `{bp['xgboost_trees']}` ·
  ARIMA order `{tuple(bp['arima_order'])}` · ANN layers `{tuple(bp['ann_layers'])}`.
""")
