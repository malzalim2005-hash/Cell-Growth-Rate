"""
==========================================================
FUNCTIONS
Data loading, model comparison, evaluation and plotting.
==========================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay

from src import constants as c


def load_data():
    df = pd.read_csv(c.DATASET_PATH)
    print(f"Loaded {len(df)} rows")
    print(df.describe().round(3))
    return df


def split_data(df):
    X = df[c.FEATURES]
    y = df[c.TARGET]
    return train_test_split(X, y, test_size=c.TEST_SIZE, random_state=c.RANDOM_STATE)


def build_candidates():
    models = {}
    for degree in c.POLY_DEGREES:
        models[f"Polynomial (degree {degree})"] = make_pipeline(
            StandardScaler(), PolynomialFeatures(degree), LinearRegression()
        )
    for k in c.KNN_NEIGHBORS:
        models[f"KNN (k = {k})"] = make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=k)
        )
    return models


def compare_models(models, X_train, y_train):
    cv = KFold(n_splits=c.N_FOLDS, shuffle=True, random_state=c.RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["neg_root_mean_squared_error", "r2"],
            return_train_score=True,
        )
        rows.append({
            "Model": name,
            "Train RMSE": -scores["train_neg_root_mean_squared_error"].mean(),
            "CV RMSE": -scores["test_neg_root_mean_squared_error"].mean(),
            "CV RMSE Std": scores["test_neg_root_mean_squared_error"].std(),
            "CV R2": scores["test_r2"].mean(),
        })
    results = pd.DataFrame(rows).sort_values("CV RMSE").round(4)
    results.to_csv(c.TABLES_DIR / "model_comparison.csv", index=False)
    print(results.to_string(index=False))
    return results


def plot_model_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    poly = results[results["Model"].str.startswith("Polynomial")].copy()
    poly["x"] = poly["Model"].str.extract(r"(\d+)").astype(int)
    poly = poly.sort_values("x")
    axes[0].plot(poly["x"], poly["Train RMSE"], "o-", label="Train")
    axes[0].plot(poly["x"], poly["CV RMSE"], "s-", label="Cross-validation")
    axes[0].set_xlabel("Polynomial degree")
    axes[0].set_title("Polynomial regression")

    knn = results[results["Model"].str.startswith("KNN")].copy()
    knn["x"] = knn["Model"].str.extract(r"(\d+)").astype(int)
    knn = knn.sort_values("x")
    axes[1].plot(knn["x"], knn["Train RMSE"], "o-", label="Train")
    axes[1].plot(knn["x"], knn["CV RMSE"], "s-", label="Cross-validation")
    axes[1].set_xlabel("Number of neighbours (k)")
    axes[1].set_title("k-nearest neighbours")

    for ax in axes:
        ax.set_ylabel("RMSE [d$^{-1}$]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(c.FIGURES_DIR / "model_comparison.png", dpi=300)
    plt.close(fig)


def evaluate_on_test(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = pd.DataFrame([{
        "Model": name,
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Test MAE": mean_absolute_error(y_test, y_pred),
        "Test R2": r2_score(y_test, y_pred),
    }]).round(4)
    metrics.to_csv(c.TABLES_DIR / "test_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", label="Perfect prediction")
    ax.set_xlabel("Measured growth rate [d$^{-1}$]")
    ax.set_ylabel("Predicted growth rate [d$^{-1}$]")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(c.FIGURES_DIR / "parity_plot.png", dpi=300)
    plt.close(fig)
    return model


def plot_feature_effects(model, X_train):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    PartialDependenceDisplay.from_estimator(model, X_train, c.FEATURES, ax=axes.ravel())
    fig.suptitle("Effect of each variable on predicted growth rate")
    fig.tight_layout()
    fig.savefig(c.FIGURES_DIR / "feature_effects.png", dpi=300)
    plt.close(fig)
