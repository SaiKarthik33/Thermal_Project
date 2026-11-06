"""
ML Surrogate Model for Heat Sink Thermal Analysis
Targets: R_total, T_junction
Models: Linear Regression, Random Forest (300 trees)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load dataset ---
df = pd.read_csv("heat_sink_dataset.csv")
features = ["TDP_W", "V_air_ms", "k_tim_WmK"]
targets  = ["R_total", "T_junction_C"]

X = df[features].values
results = []
rf_trained = {}

fig_pva, axes = plt.subplots(2, 2, figsize=(10, 8))
fig_pva.suptitle("Predicted vs Actual")

for t_idx, target in enumerate(targets):
    y = df[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }

    for m_idx, (name, model) in enumerate(models.items()):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        if name == "Random Forest":
            rf_trained[target] = model

        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2   = r2_score(y_te, y_pred)
        results.append({"Target": target, "Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"{target} | {name:20s} | MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        ax = axes[t_idx][m_idx]
        ax.scatter(y_te, y_pred, s=5, alpha=0.4)
        lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{target}\n{name} (R²={r2:.4f})")

plt.tight_layout()
fig_pva.savefig("predicted_vs_actual.png", dpi=150)
plt.close(fig_pva)

# --- Save evaluation results ---
res_df = pd.DataFrame(results)
res_df.to_csv("model_evaluation_results.csv", index=False)

# --- Correlation heatmap ---
corr = df[features + targets].corr()
fig_c, ax_c = plt.subplots(figsize=(6, 5))
im = ax_c.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(im, ax=ax_c)
labels = features + targets
ax_c.set_xticks(range(len(labels))); ax_c.set_xticklabels(labels, rotation=45, ha="right")
ax_c.set_yticks(range(len(labels))); ax_c.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax_c.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax_c.set_title("Correlation Heatmap")
plt.tight_layout()
fig_c.savefig("correlation_heatmap.png", dpi=150)
plt.close(fig_c)

# --- Feature importance (reusing trained Random Forest models) ---
fig_fi, axes_fi = plt.subplots(1, 2, figsize=(10, 4))
for ax, target, title in zip(axes_fi,
                             ["R_total", "T_junction_C"],
                             ["R_total Feature Importance", "T_junction Feature Importance"]):
    imp = rf_trained[target].feature_importances_
    ax.barh(features, imp)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    for i, v in enumerate(imp):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
plt.tight_layout()
fig_fi.savefig("feature_importance.png", dpi=150)
plt.close(fig_fi)

print("\nOutputs saved: model_evaluation_results.csv, correlation_heatmap.png, "
      "feature_importance.png, predicted_vs_actual.png")
