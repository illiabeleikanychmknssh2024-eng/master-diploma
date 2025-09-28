#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heart Disease ML Pipeline — leakage-free CV + SHAP + Ensemble (soft voting)

- 10-fold Stratified CV.
- Resampling (SMOTE/SMOTE-ENN) інтегровано в imblearn.Pipeline (без leakage).
- Масштабування StandardScaler, кодування OneHot/Ordinal в ColumnTransformer.
- Тюнінг (Optuna/Grid) над ПАЙПЛАЙНОМ; префікс параметрів 'model__'.
- SHAP для RF/XGB/LGBM + KernelExplainer для ансамблю.
-
"""

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

import joblib
import matplotlib.pyplot as plt

def _try_import(mod):
    try:
        return __import__(mod)
    except Exception:
        return None

optuna = _try_import("optuna")
xgb_mod = _try_import("xgboost")
lgbm_mod = _try_import("lightgbm")
shap = _try_import("shap")

XGB_OK = xgb_mod is not None
LGBM_OK = lgbm_mod is not None
OPTUNA_OK = optuna is not None
SHAP_OK = shap is not None

if XGB_OK:
    from xgboost import XGBClassifier
if LGBM_OK:
    from lightgbm import LGBMClassifier
if OPTUNA_OK:
    from optuna.samplers import TPESampler

RENAME_MAP = {
    "age": "Вік (age)",
    "sex": "Стать (sex)",
    "trestbps": "Тиск у спокої (trestbps)",
    "chol": "Холестерин (chol)",
    "fbs": "Вис. цукор натще (fbs)",
    "restecg_0": "restecg: 0",
    "restecg_1": "restecg: 1",
    "restecg_2": "restecg: 2",
    "thalach": "Макс. ЧСС (thalach)",
    "exang": "Стенокардія при навантаженні (exang)",
    "oldpeak": "ST депресія (oldpeak)",
    "slope_0": "slope: 0",
    "slope_1": "slope: 1",
    "slope_2": "slope: 2",
    "ca": "К-ть уражених судин (ca)",
    "thal_0": "thal: 0",
    "thal_1": "thal: 1",
    "thal_2": "thal: 2",
    "thal_3": "thal: 3",
    "cp_0": "cp: тип 0",
    "cp_1": "cp: тип 1",
    "cp_2": "cp: тип 2",
    "cp_3": "cp: тип 3",
}
APPLY_RENAME_MAP = True

# ---------- utils ----------
def results_dir() -> Path:
    out = Path(__file__).parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out

def guess_target(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in ("target", "class", "label", "disease", "hd", "outcome", "heartdisease"):
            return c
    last = df.columns[-1]
    if df[last].nunique() == 2:
        return last
    raise ValueError("Не знайдено цільову колонку. Вкажіть --target-col.")

def split_cols(df: pd.DataFrame, target_col: str):
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols, encoding: str):
    num_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    if encoding == "onehot":
        cat_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    elif encoding == "ordinal":
        cat_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])
    else:
        raise ValueError("encoding має бути 'onehot' або 'ordinal'.")

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # дає age, cp_0, ... замість довгих назв
    )
    return pre

def build_pipeline(model, imbalance: str, preprocessor, random_state: int):
    steps = [("pre", preprocessor)]
    if imbalance == "smote":
        steps.append(("resample", SMOTE(random_state=random_state, k_neighbors=5)))
    elif imbalance == "smoteenn":
        steps.append(("resample", SMOTEENN(
            random_state=random_state,
            smote=SMOTE(random_state=random_state, k_neighbors=5)
        )))
    steps.append(("model", model))
    return Pipeline(steps)

def metrics_dict(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1, "roc_auc": auc}

def plot_cm(cm, out_file: Path, labels=("0","1")):
    fig, ax = plt.subplots(figsize=(4,3.5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label", title="Confusion matrix")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout(); fig.savefig(out_file, dpi=160, bbox_inches="tight"); plt.close(fig)

def plot_roc(y_true, y_prob, out_file: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4,3.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(out_file, dpi=160, bbox_inches="tight"); plt.close(fig)

def time_inference(model, X, n_repeat=5, batch=False):
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        if batch:
            _ = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
        else:
            for i in range(min(200, X.shape[0])):
                xi = X[i:i+1]
                _ = model.predict_proba(xi) if hasattr(model, "predict_proba") else model.predict(xi)
        times.append(time.perf_counter() - start)
    return float(np.median(times))

def get_models(class_weight=False, random_state=42):
    models = {}
    lr_params = dict(max_iter=1000, random_state=random_state, solver="lbfgs")
    if class_weight:
        lr_params["class_weight"] = "balanced"
    models["logreg"] = LogisticRegression(**lr_params)

    rf_params = dict(n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state)
    if class_weight:
        rf_params["class_weight"] = "balanced"
    models["rf"] = RandomForestClassifier(**rf_params)

    if XGB_OK:
        models["xgb"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1,
            objective="binary:logistic", eval_metric="auc"
        )

    if LGBM_OK:
        models["lgbm"] = LGBMClassifier(
            n_estimators=500, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=random_state, n_jobs=-1
        )
    return models

def grids_for(name):
    if name == "logreg":
        return {"model__C": [0.01, 0.1, 1.0, 10.0], "model__penalty": ["l2"]}
    if name == "rf":
        return {"model__n_estimators": [200, 400, 800], "model__max_depth": [None, 6, 10, 16]}
    if name == "xgb":
        return {"model__n_estimators": [200, 400, 800], "model__max_depth": [3, 4, 6, 8],
                "model__learning_rate": [0.03, 0.1, 0.2],
                "model__subsample": [0.7, 0.9, 1.0], "model__colsample_bytree": [0.7, 0.9, 1.0]}
    if name == "lgbm":
        return {"model__n_estimators": [300, 600, 900], "model__num_leaves": [31, 63, 127],
                "model__learning_rate": [0.03, 0.05, 0.1]}
    return {}

def optuna_objective_factory(model_name, pipeline_base, X, y, cv):
    def obj(trial):
        pipe = clone(pipeline_base)
        if model_name == "logreg":
            pipe.set_params(model__C=trial.suggest_float("C", 1e-3, 10.0, log=True))
        elif model_name == "rf":
            pipe.set_params(
                model__n_estimators=trial.suggest_int("n_estimators", 200, 800),
                model__max_depth=trial.suggest_int("max_depth", 4, 20)
            )
        elif model_name == "xgb" and XGB_OK:
            pipe.set_params(
                model__n_estimators=trial.suggest_int("n_estimators", 200, 800),
                model__max_depth=trial.suggest_int("max_depth", 2, 10),
                model__learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                model__subsample=trial.suggest_float("subsample", 0.6, 1.0),
                model__colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0)
            )
        elif model_name == "lgbm" and LGBM_OK:
            pipe.set_params(
                model__n_estimators=trial.suggest_int("n_estimators", 300, 900),
                model__num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
                model__learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            )
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()
    return obj

def fit_with_tuning(model_name, base_pipeline, X, y, tuner="optuna", cv_splits=10, random_state=42):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    if tuner == "grid":
        grid = grids_for(model_name)
        if not grid:
            base_pipeline.fit(X, y); return base_pipeline, None
        gs = GridSearchCV(base_pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_
    elif tuner == "optuna":
        if not OPTUNA_OK:
            raise RuntimeError("Optuna не встановлено; використайте --tuner grid")
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
        study.optimize(optuna_objective_factory(model_name, base_pipeline, X, y, cv), n_trials=30, show_progress_bar=False)
        best_params = study.best_params
        best_pipe = clone(base_pipeline)
        for k, v in best_params.items():
            best_pipe.set_params(**{f"model__{k}": v})
        best_pipe.fit(X, y)
        return best_pipe, {f"model__{k}": v for k, v in best_params.items()}
    else:
        base_pipeline.fit(X, y); return base_pipeline, None

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Heart Disease Pipeline (leakage-free CV + SHAP)")
    parser.add_argument("--csv", type=str, default="heart.csv", help="Шлях до CSV (default: heart.csv поруч із скриптом).")
    parser.add_argument("--target-col", type=str, default=None, help="Назва цільової колонки (якщо None — авто).")
    parser.add_argument("--encoding", type=str, default="onehot", choices=["onehot", "ordinal"])
    parser.add_argument("--imbalance", type=str, default="smoteenn", choices=["none", "class_weight", "smote", "smoteenn"])
    parser.add_argument("--tuner", type=str, default="optuna", choices=["grid", "optuna", "none"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = results_dir()

    # Load
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV не знайдено: {args.csv}. Покладіть heart.csv поруч або вкажіть --csv шлях.")
    df = pd.read_csv(csv_path)
    target = args.target_col or guess_target(df)
    Xdf, y, num_cols, cat_cols = split_cols(df, target)

    # Split (без leakage)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        Xdf, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # Preprocessor і моделі
    pre = build_preprocessor(num_cols, cat_cols, args.encoding)
    class_weight_flag = (args.imbalance == "class_weight")
    models = get_models(class_weight=class_weight_flag, random_state=args.random_state)

    results = {}
    trained = {}
    feature_names = None
    raw_names = None

    # Train/tune кожну модель як pipeline(pre -> [resample] -> model)
    for name, model in models.items():
        base_pipe = build_pipeline(model, args.imbalance, pre, args.random_state)
        m_fitted, best_params = fit_with_tuning(
            name, base_pipe, X_train_df, y_train, tuner=args.tuner,
            cv_splits=10, random_state=args.random_state
        )

        try:
            _raw = m_fitted.named_steps["pre"].get_feature_names_out().tolist()
            _clean = [n.replace("num__", "").replace("cat__", "").replace("onehot__", "").replace("ord__", "") for n in _raw]
            raw_names = _raw
            if feature_names is None:
                feature_names = _clean
        except Exception:
            pass

        # Predict on test
        if hasattr(m_fitted, "predict_proba"):
            y_prob = m_fitted.predict_proba(X_test_df)[:, 1]
        else:
            raw = m_fitted.decision_function(X_test_df)
            y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        m = metrics_dict(y_test, y_pred, y_prob)

        # Save pipeline та latency (на трансформованих ознаках)
        model_path = out_dir / f"{name}.joblib"
        joblib.dump(m_fitted, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)

        Xt_test = m_fitted.named_steps["pre"].transform(X_test_df)
        est = m_fitted.named_steps["model"]

        def _predict_wrapper(M, X):
            return M.predict_proba(X) if hasattr(M, "predict_proba") else M.predict(X)

        # single latency
        t_single = []
        for _ in range(5):
            start = time.perf_counter()
            for i in range(min(200, Xt_test.shape[0])):
                _ = _predict_wrapper(est, Xt_test[i:i+1])
            t_single.append(time.perf_counter() - start)
        single_lat = float(np.median(t_single))
        # batch latency
        t_batch = []
        for _ in range(5):
            start = time.perf_counter()
            _ = _predict_wrapper(est, Xt_test)
            t_batch.append(time.perf_counter() - start)
        batch_lat = float(np.median(t_batch))

        m.update({
            "model_size_mb": round(size_mb, 4),
            "latency_single_median_s": round(single_lat, 6),
            "latency_batch_median_s": round(batch_lat, 6),
            "best_params": best_params
        })
        results[name] = m
        trained[name] = m_fitted

        # Plots
        cm = confusion_matrix(y_test, y_pred)
        plot_cm(cm, out_dir / f"{name}_cm.png")
        plot_roc(y_test, y_prob, out_dir / f"{name}_roc.png")

    # Ensemble (soft voting)
    ens_members = []
    for k in ("xgb", "lgbm", "rf", "logreg"):
        if k in trained:
            est = trained[k].named_steps["model"]
            ens_members.append((k, est))
    if len(ens_members) >= 2:
        voting = VotingClassifier(estimators=ens_members, voting="soft", n_jobs=-1)
        ens_pipe = build_pipeline(voting, args.imbalance, pre, args.random_state)
        ens_pipe.fit(X_train_df, y_train)
        y_prob = ens_pipe.predict_proba(X_test_df)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = metrics_dict(y_test, y_pred, y_prob)

        model_path = out_dir / "ensemble_soft.joblib"
        joblib.dump(ens_pipe, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)

        Xt_test = ens_pipe.named_steps["pre"].transform(X_test_df)
        est_ens = ens_pipe.named_steps["model"]

        def _pw(M, X): return M.predict_proba(X)
        t_single = []
        for _ in range(5):
            start = time.perf_counter()
            for i in range(min(200, Xt_test.shape[0])):
                _ = _pw(est_ens, Xt_test[i:i+1])
            t_single.append(time.perf_counter() - start)
        t_batch = []
        for _ in range(5):
            start = time.perf_counter()
            _ = _pw(est_ens, Xt_test)
            t_batch.append(time.perf_counter() - start)

        m.update({
            "model_size_mb": round(size_mb, 4),
            "latency_single_median_s": round(float(np.median(t_single)), 6),
            "latency_batch_median_s": round(float(np.median(t_batch)), 6),
            "members": [k for k, _ in ens_members]
        })
        results["ensemble_soft"] = m

        cm = confusion_matrix(y_test, y_pred)
        plot_cm(cm, out_dir / "ensemble_soft_cm.png")
        plot_roc(y_test, y_prob, out_dir / "ensemble_soft_roc.png")

    # ---------- XAI (SHAP) ----------
    xai = {}
    if SHAP_OK:
        for nm in ["xgb", "lgbm", "rf"]:
            if nm in trained:
                try:
                    pipe = trained[nm]
                    model = pipe.named_steps["model"]
                    Xt_test = pipe.named_steps["pre"].transform(X_test_df)

                    rng = np.random.RandomState(42)
                    idx = rng.choice(np.arange(Xt_test.shape[0]), size=min(200, Xt_test.shape[0]), replace=False)
                    Xs = Xt_test[idx]

                    fnames = feature_names if feature_names is not None else (
                        raw_names if raw_names is not None else [f"f{i}" for i in range(Xs.shape[1])]
                    )
                    Xs_df = pd.DataFrame(Xs, columns=fnames)
                    if APPLY_RENAME_MAP:
                        Xs_df = Xs_df.rename(columns=RENAME_MAP)

                    explainer = shap.TreeExplainer(model, model_output="probability")
                    shap_vals = explainer.shap_values(Xs_df)

                    mean_abs = (np.mean(np.abs(shap_vals), axis=0)).tolist()
                    xai[nm] = {"mean_abs_shap": mean_abs, "feature_names_used": list(Xs_df.columns)[:10]}

                    plt.figure(); shap.summary_plot(shap_vals, Xs_df, show=False)
                    plt.tight_layout(); plt.savefig(out_dir / f"{nm}_shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()

                    plt.figure(); shap.summary_plot(shap_vals, Xs_df, plot_type="bar", show=False)
                    plt.tight_layout(); plt.savefig(out_dir / f"{nm}_shap_bar.png", dpi=160, bbox_inches="tight"); plt.close()
                except Exception as e:
                    xai[nm] = {"error": str(e)}

        # ансамбль (KernelExplainer)
        if "ensemble_soft" in results and len(ens_members) >= 2:
            try:
                ens_pipe = joblib.load(out_dir / "ensemble_soft.joblib")
                rng = np.random.RandomState(42)
                Xt_train = ens_pipe.named_steps["pre"].transform(X_train_df)
                Xt_test = ens_pipe.named_steps["pre"].transform(X_test_df)

                bg_idx = rng.choice(np.arange(Xt_train.shape[0]), size=min(100, Xt_train.shape[0]), replace=False)
                X_bg = Xt_train[bg_idx]
                xs_idx = rng.choice(np.arange(Xt_test.shape[0]), size=min(200, Xt_test.shape[0]), replace=False)
                Xs = Xt_test[xs_idx]

                fnames = feature_names if feature_names is not None else (
                    raw_names if raw_names is not None else [f"f{i}" for i in range(Xs.shape[1])]
                )
                X_bg_df = pd.DataFrame(X_bg, columns=fnames)
                Xs_df   = pd.DataFrame(Xs,   columns=fnames)
                if APPLY_RENAME_MAP:
                    X_bg_df = X_bg_df.rename(columns=RENAME_MAP)
                    Xs_df   = Xs_df.rename(columns=RENAME_MAP)

                f_ens = lambda data: ens_pipe.named_steps["model"].predict_proba(np.asarray(data))[:, 1]
                expl_ens = shap.KernelExplainer(f_ens, X_bg_df)
                shap_vals_ens = expl_ens.shap_values(Xs_df, nsamples="auto")

                xai["ensemble_soft"] = {
                    "mean_abs_shap": (np.mean(np.abs(shap_vals_ens), axis=0)).tolist(),
                    "feature_names_used": list(Xs_df.columns)[:10]
                }

                plt.figure(); shap.summary_plot(shap_vals_ens, Xs_df, show=False)
                plt.tight_layout(); plt.savefig(out_dir / "ensemble_soft_shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()

                plt.figure(); shap.summary_plot(shap_vals_ens, Xs_df, plot_type="bar", show=False)
                plt.tight_layout(); plt.savefig(out_dir / "ensemble_soft_shap_bar.png", dpi=160, bbox_inches="tight"); plt.close()
            except Exception as e:
                xai["ensemble_soft"] = {"error": str(e)}

    # Save summary JSON
    summary = {
        "target": target,
        "encoding": args.encoding,
        "imbalance": args.imbalance,
        "tuner": args.tuner,
        "preprocess": {"type": args.encoding, "column_transformer": True, "scaler": "standard"},
        "cv_folds": 10,
        "results": results,
        "xai": xai,
        "feature_names_available": feature_names is not None
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] All results saved to:", out_dir.resolve())

if __name__ == "__main__":
    main()
