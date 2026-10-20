#!/usr/bin/env python3
"""Cross-validate compact capacity-regression heads on frozen ViT features."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def candidates():
    return {
        "ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": np.logspace(-5, 5, 21)},
        ),
        "pls": (
            PLSRegression(scale=True, max_iter=1000),
            {"n_components": [2, 4, 8, 12, 16, 24, 32]},
        ),
        "svr": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
                "model__epsilon": [0.001, 0.005, 0.01, 0.03, 0.05],
                "model__gamma": ["scale", 0.0001, 0.0003, 0.001, 0.003],
            },
        ),
        "kernel_ridge": (
            Pipeline([("scale", StandardScaler()), ("model", KernelRidge(kernel="rbf"))]),
            {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "model__gamma": [0.0001, 0.0003, 0.001, 0.003],
            },
        ),
        "knn": (
            Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor())]),
            {
                "model__n_neighbors": [2, 3, 4, 5, 7, 9, 12],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "extra_trees": (
            ExtraTreesRegressor(n_estimators=400, random_state=2026, n_jobs=-1),
            {
                "max_features": [0.3, 0.6, 1.0],
                "min_samples_leaf": [1, 2, 4],
                "max_depth": [None, 8, 16],
            },
        ),
        "random_forest": (
            RandomForestRegressor(n_estimators=400, random_state=2026, n_jobs=-1),
            {
                "max_features": [0.3, 0.6, 1.0],
                "min_samples_leaf": [1, 2, 4],
                "max_depth": [None, 8, 16],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=2026),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.03, 0.1],
                "max_depth": [1, 2, 3],
                "min_samples_leaf": [1, 3, 5],
                "loss": ["squared_error", "huber"],
            },
        ),
    }


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cv-jobs", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    for feature_path in sorted(args.features_dir.glob("*_features.npz")):
        battery = feature_path.stem[: -len("_features")].upper()
        data = np.load(feature_path)
        x_train = data["x_train"].astype(np.float64)
        y_train = data["y_train"].astype(np.float64).reshape(-1)
        x_test = data["x_test"].astype(np.float64)
        y_test = data["y_test"].astype(np.float64).reshape(-1)
        print(f"\n{battery}: train={len(y_train)}, test={len(y_test)}", flush=True)

        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=2026)
        reports = []
        fitted = {}
        for name, (model, grid) in candidates().items():
            search = GridSearchCV(
                model,
                grid,
                scoring="neg_mean_absolute_error",
                cv=cv,
                n_jobs=args.cv_jobs,
                refit=True,
            )
            search.fit(x_train, y_train)
            pred = np.asarray(search.predict(x_test)).reshape(-1)
            report = {
                "name": name,
                "cv_mae": float(-search.best_score_),
                "test_r2": float(r2_score(y_test, pred)),
                "test_mae": float(mean_absolute_error(y_test, pred)),
                "test_rmse": float(mean_squared_error(y_test, pred) ** 0.5),
                "best_params": {key: scalar(value) for key, value in search.best_params_.items()},
                "predictions": pred.tolist(),
            }
            reports.append(report)
            fitted[name] = search.best_estimator_
            print(json.dumps(report, ensure_ascii=False), flush=True)

        reports.sort(key=lambda item: item["cv_mae"])
        selected = reports[0]["name"]
        joblib.dump(fitted[selected], args.output_dir / f"{battery.lower()}_head.joblib", compress=3)
        battery_report = {
            "selected_by": "lowest repeated-5-fold CV MAE on training split",
            "selected_model": selected,
            "labels": y_test.tolist(),
            "models": reports,
        }
        all_reports[battery] = battery_report
        (args.output_dir / f"{battery.lower()}_report.json").write_text(
            json.dumps(battery_report, indent=2), encoding="utf-8"
        )
        print(f"Selected {battery}: {selected}", flush=True)

    (args.output_dir / "report.json").write_text(
        json.dumps(all_reports, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
