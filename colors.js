// One fixed colour per model so every chart uses the same colours.
export const MODEL_COLORS = {
  random_forest: "#16a34a",
  gradient_boosting: "#f59e0b",
  xgboost: "#ef4444",
  voting: "#3b82f6",
  stacking: "#8b5cf6",
  hybrid_rf_arima: "#0d9488",
  hybrid_rf_ann: "#db2777",
};

export const GROUP_BADGE = {
  Single: "bg-green-100 text-green-700",
  Ensemble: "bg-blue-100 text-blue-700",
  Hybrid: "bg-pink-100 text-pink-700",
};

export const fmt = (n, d = 1) =>
  n === null || n === undefined ? "-" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d });
