#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heart Disease ML Pipeline — advisor-complete version
Скрипт виконує:
- Завантаження CSV (за замовч., файл "heart.csv" поруч із скриптом)
- Кодування: onehot | ordinal | target (для target потрібен category-encoders)
- Імпутація пропусків, масштабування числових
- Балансування: none | class_weight | smote | smoteenn
- Моделі: LogisticRegression, RandomForest, XGBoost, LightGBM
- Тюнінг: grid | optuna | none (ROC-AUC як ціль)
- Метрики на тесті: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC
- Ансамбль: soft voting
- XAI: SHAP для дерев’яних моделей (RF/XGB/LGBM)
- Аналіз ресурсів: розмір моделі (MB) та latency (single/batch)
Результати зберігаються у ./results поруч зі скриптом.
"""

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.base import clone

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

import joblib
import matplotlib.pyplot as plt

# --- опційні пакети
def _try_import(mod, name):
    try:
        return __import__(mod)
    except Exception:
        return None

optuna = _try_import("optuna", "optuna")
xgb_mod = _try_import("xgboost", "xgboost")
lgbm_mod = _try_import("lightgbm", "lightgbm")
shap = _try_import("shap", "shap")
ce = _try_import("category_encoders", "category_encoders")

XGB_OK = xgb_mod is not None
LGBM_OK = lgbm_mod is not None
OPTUNA_OK = optuna is not None
SHAP_OK = shap is not None
CE_OK = ce is not None

if XGB_OK:
    from xgboost import XGBClassifier
if LGBM_OK:
    from lightgbm import LGBMClassifier
if OPTUNA_OK:
    from optuna.samplers import TPESampler

# -------- утиліти --------
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
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    if encoding == "onehot":
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        return ColumnTransformer([("num", num_pipe, num_cols),
                                  ("cat", cat_pipe, cat_cols)])
    elif encoding == "ordinal":
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
        return ColumnTransformer([("num", num_pipe, num_cols),
                                  ("cat", cat_pipe, cat_cols)])
    elif encoding == "target":
        if not CE_OK:
            raise RuntimeError("Встановіть 'category-encoders' для target encoding: pip install category-encoders")
        # Target encoding потребує y, тож робимо вручну в main()
        return ("target", num_cols, cat_cols)
    else:
        raise ValueError("encoding має бути onehot | ordinal | target")

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
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
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
    lr_params = dict(max_iter=1000, random_state=random_state)
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
        return {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"]}
    if name == "rf":
        return {"n_estimators": [200, 400, 800], "max_depth": [None, 6, 10, 16]}
    if name == "xgb":
        return {"n_estimators": [200, 400, 800], "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.03, 0.1, 0.2],
                "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.7, 0.9, 1.0]}
    if name == "lgbm":
        return {"n_estimators": [300, 600, 900], "num_leaves": [31, 63, 127],
                "learning_rate": [0.03, 0.05, 0.1]}
    return {}

def optuna_objective_factory(model_name, base_model, X, y, cv):
    def obj(trial):
        model = clone(base_model)
        if model_name == "logreg":
            model.set_params(C=trial.suggest_float("C", 1e-3, 10.0, log=True))
        elif model_name == "rf":
            model.set_params(n_estimators=trial.suggest_int("n_estimators", 200, 800),
                             max_depth=trial.suggest_int("max_depth", 4, 20))
        elif model_name == "xgb" and XGB_OK:
            model.set_params(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0)
            )
        elif model_name == "lgbm" and LGBM_OK:
            model.set_params(
                n_estimators=trial.suggest_int("n_estimators", 300, 900),
                num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            )
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()
    return obj

def fit_with_tuning(name, model, X, y, tuner="optuna", cv_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    if tuner == "grid":
        grid = grids_for(name)
        if not grid:
            model.fit(X, y); return model, None
        gs = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_
    elif tuner == "optuna":
        if not OPTUNA_OK:
            raise RuntimeError("Optuna не встановлено; використайте --tuner grid")
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
        study.optimize(optuna_objective_factory(name, model, X, y, cv), n_trials=30, show_progress_bar=False)
        best_params = study.best_params
        model.set_params(**best_params)
        model.fit(X, y)
        return model, best_params
    else:
        model.fit(X, y); return model, None

# -------- main --------
def main():
    parser = argparse.ArgumentParser(description="Heart Disease Pipeline (advisor-complete)")
    parser.add_argument("--csv", type=str, default="heart.csv", help="Шлях до CSV (default: heart.csv поруч із скриптом).")
    parser.add_argument("--target-col", type=str, default=None, help="Назва цільової колонки (якщо None — авто).")
    parser.add_argument("--encoding", type=str, default="onehot", choices=["onehot", "ordinal", "target"])
    parser.add_argument("--imbalance", type=str, default="smoteenn", choices=["none", "class_weight", "smote", "smoteenn"])
    parser.add_argument("--tuner", type=str, default="optuna", choices=["grid", "optuna", "none"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = results_dir()

    # Load
    if not Path(args.csv).exists():
        raise FileNotFoundError(f"CSV не знайдено: {args.csv}. Покладіть heart.csv поруч або вкажіть --csv шлях.")
    df = pd.read_csv(args.csv)
    target = args.target_col or guess_target(df)
    Xdf, y, num_cols, cat_cols = split_cols(df, target)

    # Preprocess
    if args.encoding == "target":
        if not CE_OK:
            raise RuntimeError("Встановіть 'category-encoders' для target encoding: pip install category-encoders")
        cat_imp = SimpleImputer(strategy="most_frequent")
        X_cat = pd.DataFrame(cat_imp.fit_transform(Xdf[cat_cols]), columns=cat_cols, index=Xdf.index)
        te = ce.TargetEncoder(cols=cat_cols)
        X_cat_te = te.fit_transform(X_cat, y)
        num_imp = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(num_imp.fit_transform(Xdf[num_cols]), columns=num_cols, index=Xdf.index)
        scaler = StandardScaler()
        X_num_sc = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols, index=Xdf.index)
        X_proc = pd.concat([X_num_sc, X_cat_te], axis=1).values
        preprocess_info = {"type": "target", "cat_imputer": "most_frequent", "num_imputer": "median", "scaler": "standard"}
    else:
        ct = build_preprocessor(num_cols, cat_cols, args.encoding)
        X_proc = ct.fit_transform(Xdf)
        preprocess_info = {"type": args.encoding, "column_transformer": True}

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # Imbalance
    class_weight_flag = (args.imbalance == "class_weight")
    if args.imbalance == "smote":
        sm = SMOTE(random_state=args.random_state, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    elif args.imbalance == "smoteenn":
        smoteenn = SMOTEENN(random_state=args.random_state, smote=SMOTE(random_state=args.random_state, k_neighbors=5))
        X_train, y_train = smoteenn.fit_resample(X_train, y_train)

    # Models
    models = get_models(class_weight=class_weight_flag, random_state=args.random_state)

    results = {}
    trained = {}

    # Fit/tune each model
    for name, model in models.items():
        m_fitted, best_params = fit_with_tuning(name, model, X_train, y_train, tuner=args.tuner,
                                                cv_splits=5, random_state=args.random_state)

        # Predict proba
        if hasattr(m_fitted, "predict_proba"):
            y_prob = m_fitted.predict_proba(X_test)[:, 1]
        else:
            raw = m_fitted.decision_function(X_test)
            y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        m = metrics_dict(y_test, y_pred, y_prob)

        # Resource analysis
        model_path = out_dir / f"{name}.joblib"
        joblib.dump(m_fitted, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        single_lat = time_inference(m_fitted, X_test, n_repeat=5, batch=False)
        batch_lat = time_inference(m_fitted, X_test, n_repeat=5, batch=True)

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
    ens_members = [(k, v) for k, v in trained.items() if k in ("xgb","lgbm","rf","logreg")]
    if len(ens_members) >= 2:
        ens = VotingClassifier(estimators=ens_members, voting="soft", n_jobs=-1)
        ens.fit(X_train, y_train)
        y_prob = ens.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        m = metrics_dict(y_test, y_pred, y_prob)

        model_path = out_dir / "ensemble_soft.joblib"
        joblib.dump(ens, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        single_lat = time_inference(ens, X_test, n_repeat=5, batch=False)
        batch_lat = time_inference(ens, X_test, n_repeat=5, batch=True)

        m.update({
            "model_size_mb": round(size_mb, 4),
            "latency_single_median_s": round(single_lat, 6),
            "latency_batch_median_s": round(batch_lat, 6),
            "members": [k for k, _ in ens_members]
        })
        results["ensemble_soft"] = m

        cm = confusion_matrix(y_test, y_pred)
        plot_cm(cm, out_dir / "ensemble_soft_cm.png")
        plot_roc(y_test, y_prob, out_dir / "ensemble_soft_roc.png")

    # SHAP (tree models only)
    xai = {}
    if SHAP_OK:
        for nm in ["xgb","lgbm","rf"]:
            if nm in trained:
                try:
                    model = trained[nm]
                    rng = np.random.RandomState(42)
                    idx = rng.choice(np.arange(X_test.shape[0]), size=min(200, X_test.shape[0]), replace=False)
                    Xs = X_test[idx]
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(Xs)
                    # Save mean |SHAP| for report
                    xai[nm] = {"mean_abs_shap": np.mean(np.abs(shap_vals), axis=0).tolist()}
                    # Figures
                    plt.figure()
                    shap.summary_plot(shap_vals, Xs, show=False)
                    plt.tight_layout()
                    plt.savefig(out_dir / f"{nm}_shap_summary.png", dpi=160, bbox_inches="tight")
                    plt.close()
                    plt.figure()
                    shap.summary_plot(shap_vals, Xs, plot_type="bar", show=False)
                    plt.tight_layout()
                    plt.savefig(out_dir / f"{nm}_shap_bar.png", dpi=160, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    xai[nm] = {"error": str(e)}

    # Save summary JSON
    summary = {
        "target": target,
        "encoding": args.encoding,
        "imbalance": args.imbalance,
        "tuner": args.tuner,
        "preprocess": preprocess_info,
        "results": results,
        "xai": xai
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] All results saved to:", out_dir.resolve())

if __name__ == "__main__":
    main()
