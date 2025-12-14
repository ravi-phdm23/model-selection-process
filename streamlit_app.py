# streamlit_app.py
from distro import name
import streamlit as st
# Prefer wide layout for the app by default
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np  # Added for benchmark calculations
import altair as alt
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from io import StringIO # Added for reading full DF
from dotenv import load_dotenv # --- NEW: Import dotenv ---
from feature_importance_paper import compute_feature_importance_for_files
import hashlib, io
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
import shutil
try:
    import openai
except ImportError:
    openai = None
import re
from expand_controls import (
    EXPAND_ALL_FLAG,
    queue_expand_all,
    fire_expand_all_if_pending,
    render_generate_report_button,
)
import inspect

# Optional statistical tests
try:
    from scipy import stats
    from scipy.stats import ks_2samp, mannwhitneyu, norm, chi2
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    stats = None
    ks_2samp = None
    mannwhitneyu = None
    norm = None
    chi2 = None

# Logistic regression for diagnostics
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_LR_AVAILABLE = True
except Exception:
    SKLEARN_LR_AVAILABLE = False
    LogisticRegression = None

# NEW imports from shap_analysis.py
from shap_analysis import (
    shap_rank_stability, model_randomization_sanity,
    find_counterfactual, CFConstraints,
    shap_top_interactions_for_tree, plot_ice_pdp,
    compute_single_row_reliability
)

# Import ResultManager for centralized file saving
try:
    from result_manager import get_result_manager
    RESULT_MANAGER_AVAILABLE = True
except ImportError:
    RESULT_MANAGER_AVAILABLE = False
    get_result_manager = None


def run_reliability_diagnostics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Given a DataFrame with columns: reliability_score, pred_default, actual_default,
    compute statistical tests that evaluate how well reliability_score separates
    correct vs incorrect predictions.

    Returns a dict with keys:
      - ks_stat, ks_pvalue
      - mw_stat, mw_pvalue
      - roc_auc
      - logit_coef, logit_pvalue, logit_pseudo_r2
      - n_correct, n_wrong
    """
    # Validate columns
    required = {"reliability_score", "pred_default", "actual_default"}
    if not required.issubset(set(results_df.columns)):
        missing = required.difference(set(results_df.columns))
        raise ValueError(f"results_df is missing required columns: {missing}")

    # Drop NA rows in required columns
    df = results_df.loc[:, ["reliability_score", "pred_default", "actual_default"]].dropna()
    if df.empty:
        raise ValueError("No valid rows after dropping NA in required columns.")

    scores = df["reliability_score"].astype(float)
    correct = (df["pred_default"] == df["actual_default"]).astype(int)

    scores_correct = scores[correct == 1]
    scores_wrong = scores[correct == 0]

    out: Dict[str, Any] = {}
    out['n_correct'] = int((correct == 1).sum())
    out['n_wrong'] = int((correct == 0).sum())

    # KS test and Mann-Whitney (if scipy available)
    if SCIPY_AVAILABLE:
        try:
            ks_stat, ks_p = ks_2samp(scores_correct, scores_wrong)
        except Exception:
            ks_stat, ks_p = None, None
        try:
            mw_stat, mw_p = mannwhitneyu(scores_correct, scores_wrong, alternative='greater')
        except Exception:
            mw_stat, mw_p = None, None
    else:
        ks_stat = ks_p = mw_stat = mw_p = None

    out['ks_stat'] = ks_stat
    out['ks_pvalue'] = ks_p
    out['mw_stat'] = mw_stat
    out['mw_pvalue'] = mw_p

    # ROC AUC for separating correct vs wrong using reliability_score
    try:
        roc_auc = float(roc_auc_score(correct, scores))
    except Exception:
        roc_auc = None
    out['roc_auc'] = roc_auc

    # Logistic regression: correctness ~ reliability_score
    try:
        y = correct.values
        X = scores.values.reshape(-1, 1)
        if SKLEARN_LR_AVAILABLE:
            lr = LogisticRegression(solver='lbfgs', max_iter=1000)
            lr.fit(X, y)
            coef = float(lr.coef_[0][0])
            out['logit_coef'] = coef

            # Predicted probabilities
            p_hat = lr.predict_proba(X)[:, 1]

            # Log-likelihoods
            eps = 1e-12
            p_hat_clipped = np.clip(p_hat, eps, 1 - eps)
            loglik_full = float((y * np.log(p_hat_clipped) + (1 - y) * np.log(1 - p_hat_clipped)).sum())
            p_null = float(y.mean())
            p_null = np.clip(p_null, eps, 1 - eps)
            loglik_null = float((y * np.log(p_null) + (1 - y) * np.log(1 - p_null)).sum())

            # Likelihood ratio test
            try:
                LR = 2.0 * (loglik_full - loglik_null)
                p_lr = float(chi2.sf(LR, df=1)) if chi2 is not None else None
            except Exception:
                LR = None
                p_lr = None

            out['logit_pvalue'] = p_lr

            # McFadden's pseudo-R2
            try:
                pseudo_r2 = 1.0 - (loglik_full / loglik_null) if loglik_null != 0 else None
            except Exception:
                pseudo_r2 = None
            out['logit_pseudo_r2'] = pseudo_r2

            # Try to estimate coefficient p-value via Wald test using observed Fisher information
            try:
                X_design = np.column_stack([np.ones(len(X)), X])
                W = p_hat * (1 - p_hat)
                # Fisher information approx: X^T W X
                XT_W = (X_design * W[:, None]).T
                fisher = XT_W.dot(X_design)
                cov = np.linalg.inv(fisher)
                se = float(np.sqrt(np.abs(cov[1, 1])))
                z = coef / (se + 1e-18)
                p_wald = float(2.0 * (1.0 - norm.cdf(abs(z)))) if norm is not None else None
            except Exception:
                p_wald = None

            # Prefer Wald p-value if available, otherwise LR p-value
            out['logit_pvalue_wald'] = p_wald
            if out['logit_pvalue'] is None and p_wald is not None:
                out['logit_pvalue'] = p_wald
        else:
            out['logit_coef'] = None
            out['logit_pvalue'] = None
            out['logit_pseudo_r2'] = None
            out['logit_pvalue_wald'] = None
    except Exception:
        out['logit_coef'] = None
        out['logit_pvalue'] = None
        out['logit_pseudo_r2'] = None
        out['logit_pvalue_wald'] = None

    return out


# --- NEW: Load .env file ---
# Make sure .env is in the same directory as streamlit_app.py
load_dotenv()

from io import BytesIO

# --- YData/Pandas Profiling support ---
try:
    from ydata_profiling import ProfileReport  # preferred
except ImportError:
    try:
        from pandas_profiling import ProfileReport  # legacy fallback
    except ImportError:
        ProfileReport = None  # the UI will warn and disable profiling

# Import the MODELS dictionary and the new run_experiment function
try:
    from models import MODELS, run_experiment, IMBLEARN_AVAILABLE
except ImportError:
    st.error("Could not find 'MODELS' dictionary, 'run_experiment' function, or 'IMBLEARN_AVAILABLE' in models.py. Please ensure they are defined.")
    MODELS = {}
    IMBLEARN_AVAILABLE = False
    def run_experiment(files, target, models, use_smote): # Added use_smote
        return {"error": "models.py not found"}

# --- MODIFIED: Import BOTH SHAP functions ---
try:
    import shap
    import matplotlib.pyplot as plt # Import matplotlib
    
    # Load the SHAP JavaScript libraries (for waterfall plot)
    shap.initjs()
    
    # Import both global and local SHAP functions
    from shap_analysis import get_shap_values, get_local_shap_explanation, summarize_reliability, get_shap_values_stable
    # --- NEW: Import LLM explanation function ---
    from llm_explain import get_llm_explanation
    
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    get_shap_values = None 
    get_local_shap_explanation = None
    get_llm_explanation = None # Add placeholder
    st.warning(f"A required library was not found. Steps 5 & 6 may be disabled. Error: {e}")


# ---- Session bootstrap: guarantee keys exist even before __init__ runs ----
_DEFAULTS = {
    "uploaded_files_map": {},
    "selected_datasets": [],
    "use_smote": False,
    "target_column": "target",
    "selected_model_groups": [],
    "selected_models": [],
    "results": {},
    "benchmark_results_df": None,
    "benchmark_auc_comparison": None,
    "run_shap": False,
    "full_dfs": {},
    "feature_selection": {},
    "fi_results_cache": {},
    "fi_signature": None,
    "fi_stale": False,
    "benchmark_requested": False,
    "ydata_profiles": {},         # { dataset_name: {"html": str, "filename": str} }
    "ydata_minimal_mode": False,  # remember the toggle choice
    "rel_n_trials": 1,
    "rel_n_bg": 20,
    "rel_speed_preset": "Minimal (1 trial / 20 bg) - Fastest",
    "_rel_prev_preset": "Minimal (1 trial / 20 bg) - Fastest",
    "use_stable_shap": False,
    "shap_selected_datasets": [],
    "stable_shap_trials": 2,
    "stable_shap_bg_size": 50,
    "stable_shap_explain_size": 50,
    # Reliability step defaults
    "run_reliability_test": False,
    "reliability_selected_datasets": [],
    "reliability_results": {},
    "reliability_timestamps": {},
    "reliability_ratios": {},
    "reliability_texts": {},
    # Mapping of dataset -> saved figure paths (bar/dot/waterfalls/pdp)
    "global_shap_figures": {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _df_to_named_bytesio(df, out_name: str) -> BytesIO:
    data = df.to_csv(index=False).encode("utf-8")
    bio = BytesIO(data)
    bio.seek(0)
    bio.name = out_name
    return bio


def _reset_experiment_state():
    # Clear ONLY things produced by Step 3/4 so the user must re-run.
    for k in [
        "results",
        "benchmark_results_df",
        "benchmark_auc_comparison",
        "trained_models",
        "cv_reports",
        "run_shap",
        "full_dfs",
    ]:
        st.session_state.pop(k, None)


def _df_to_named_bytesio(df, out_name: str) -> BytesIO:
    data = df.to_csv(index=False).encode("utf-8")  # produce bytes
    bio = BytesIO(data)
    bio.seek(0)
    bio.name = out_name
    return bio


def _prepare_benchmark_long(bench_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the benchmark dataframe is in long/tidy form with columns:
    ['dataset','model','metric','value']
    Accepts a wide-form `bench_df` or a pre-existing long-form and returns long-form.
    """
    if bench_df is None or bench_df.empty:
        return pd.DataFrame(columns=["dataset", "model", "metric", "value"])

    df = bench_df.copy()

    # identify dataset and model columns (common names)
    dataset_col = None
    model_col = None
    
    # First, check if we have a "Model Group" column - prefer this for the model
    for c in df.columns:
        if c.lower() in ("model group", "model_group", "modelgroup"):
            model_col = c
            break
    
    # If no Model Group column, try other model column names
    if model_col is None:
        for c in df.columns:
            if c.lower() in ("model", "model_name", "benchmark model", "benchmark_model"):
                model_col = c
                break
    
    for c in df.columns:
        if c.lower() in ("dataset", "dataset_name", "datasetname"):
            dataset_col = c
            break

    # fallback guesses
    if dataset_col is None:
        if "Dataset" in df.columns:
            dataset_col = "Dataset"
        else:
            dataset_col = df.columns[0]
    if model_col is None:
        # try common patterns
        for guess in ("Model Group", "Model", "model", "Benchmark Model", "benchmark_model", "BenchmarkModel"):
            if guess in df.columns:
                model_col = guess
                break
        if model_col is None:
            # fallback to second column if available
            model_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Desired metrics (use these names for charts/tables)
    desired = ["AUC", "PCC", "F1", "Recall", "BS", "KS", "PG", "H"]

    # Collect available metrics from the DataFrame
    available_metrics = [c for c in df.columns if c in desired]

    # Now melt using only desired metrics that are present
    metric_candidates = [m for m in desired if m in df.columns]
    if not metric_candidates:
        # Try common alternate names / uppercase matches
        desired_upper = tuple([d.upper() for d in desired])
        metric_candidates = [c for c in df.columns if c.upper() in desired_upper]

    if not metric_candidates:
        # Nothing to plot
        return pd.DataFrame(columns=["dataset", "model", "metric", "value"])

    df_long = df.melt(id_vars=[dataset_col, model_col], value_vars=metric_candidates, var_name="metric", value_name="value")
    df_long = df_long.rename(columns={dataset_col: "dataset", model_col: "model"})
    # Normalize model names to simple labels (if needed)
    df_long["model"] = df_long["model"].astype(str)
    
    # If model_col was "Model Group", the values are already group labels, no mapping needed
    # Otherwise, try to map model names to canonical group labels
    model_col_lower = model_col.lower() if isinstance(model_col, str) else ""
    is_already_group = "model group" in model_col_lower or "model_group" in model_col_lower or "modelgroup" in model_col_lower
    
    if not is_already_group:
        # Map model keys to canonical group labels (no guessing) — use exact list
        # Order matters: put more specific group labels before shorter ones
        canonical_groups = [
            "lr_reg",
            "lr",
            "adaboost",
            "Bag-CART",
            "BagNN",
            "Boost-DT",
            "RF",
            "SGB",
            "KNN",
            "XGB",
            "LGBM",
            "DL",
        ]

        def _map_to_group_label(s):
            try:
                s_str = str(s)
            except Exception:
                return s
            for g in canonical_groups:
                try:
                    if g.lower() in s_str.lower():
                        return g
                except Exception:
                    continue
            # fallback: return original model string
            return s_str

        df_long["model"] = df_long["model"].apply(_map_to_group_label)
    
    return df_long[["dataset", "model", "metric", "value"]]


def make_metric_chart(df_long: pd.DataFrame, metric_name: str):
    """
    df_long contains columns: ['dataset','model','metric','value']
    Returns an Altair chart for one metric.
    """
    # Fixed model order (use the canonical group labels you provided)
    model_order = ["lr","lr_reg","adaboost","Bag-CART","BagNN","Boost-DT","RF","SGB","KNN","XGB","LGBM","DL"]
    d = df_long[df_long["metric"] == metric_name].copy()
    if d.empty:
        # return an empty chart placeholder
        return alt.Chart(pd.DataFrame({"model":[], "value":[], "dataset":[]})).mark_line().encode()

    # Ensure model is categorical with fixed domain so x-axis order is consistent
    d["model"] = pd.Categorical(d["model"], categories=model_order, ordered=True)

    chart = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=alt.X("model:N", sort=model_order, title="Model"),
            # Show y-axis labels with four decimal places and avoid forcing zero baseline
            y=alt.Y(
                "value:Q",
                title=metric_name,
                axis=alt.Axis(format=".4f", tickCount=5),
                scale=alt.Scale(zero=False, nice=False),
            ),
            color=alt.Color("dataset:N", legend=alt.Legend(title="Dataset", orient="bottom")),
            tooltip=[alt.Tooltip("dataset:N"), alt.Tooltip("model:N"), alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=250)
    )
    return chart


def _wilcoxon_abs_error_test(y_true, p1, p2):
    """Wilcoxon on per-sample absolute error; returns (stat, p, med_diff)."""
    if not SCIPY_AVAILABLE:
        return None, None, None
    y_true = np.asarray(y_true).astype(float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if y_true.shape[0] != p1.shape[0] or p1.shape[0] != p2.shape[0]:
        return None, None, None
    diff = np.abs(p1 - y_true) - np.abs(p2 - y_true)
    if np.allclose(diff, 0):
        return 0.0, 1.0, 0.0
    try:
        stat, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p), float(np.median(diff))
    except Exception:
        return None, None, None


def _mcnemar_test(y_true, pred1, pred2):
    """McNemar's test with continuity correction; returns (b, c, chi2, p)."""
    if not SCIPY_AVAILABLE:
        return None, None, None, None
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(pred1).astype(int)
    p2 = np.asarray(pred2).astype(int)
    if y_true.shape[0] != p1.shape[0] or p1.shape[0] != p2.shape[0]:
        return None, None, None, None
    b = int(np.sum((p1 == y_true) & (p2 != y_true)))
    c = int(np.sum((p1 != y_true) & (p2 == y_true)))
    if b + c == 0:
        return b, c, None, None
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = float(stats.chi2.sf(chi2, df=1))
    return b, c, float(chi2), p_val


def _compute_midrank(x):
    x = np.asarray(x)
    idx = np.argsort(x)
    sorted_x = x[idx]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        ranks[i:j] = mid
        i = j
    out = np.empty(len(x), dtype=float)
    out[idx] = ranks
    return out


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong algorithm; predictions_sorted_transposed shape = (n_classifiers, n_examples)
    label_1_count is #positives after sorting by first classifier descending.
    """
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])

    tz = _compute_midrank(predictions_sorted_transposed[0])
    aucs = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)

    v10 = (tz[:m] - tx[0]) / n
    v01 = 1.0 - (tz[m:] - ty[0]) / m
    sx = np.cov(v10)
    sy = np.cov(v01)
    s = sx / m + sy / n
    return np.array([aucs]) if np.isscalar(aucs) else aucs, s


def delong_roc_test(y_true, scores1, scores2):
    """Returns dict with AUCs, diff, variance, z, p_value using DeLong."""
    if not SCIPY_AVAILABLE:
        return None
    y_true = np.asarray(y_true).astype(int)
    scores1 = np.asarray(scores1, dtype=float)
    scores2 = np.asarray(scores2, dtype=float)
    if y_true.ndim != 1 or scores1.shape[0] != y_true.shape[0] or scores2.shape[0] != y_true.shape[0]:
        return None
    if len(np.unique(y_true)) < 2:
        return None

    order = np.argsort(-scores1)
    labels_sorted = y_true[order]
    preds_sorted = np.vstack([scores1, scores2])[:, order]
    label_1_count = int(labels_sorted.sum())
    aucs, cov = _fast_delong(preds_sorted, label_1_count)

    # If DeLong collapses to a single AUC (degenerate), fall back to simple AUCs without p-value
    if np.asarray(aucs).shape[0] < 2:
        try:
            auc1 = float(roc_auc_score(y_true, scores1))
            auc2 = float(roc_auc_score(y_true, scores2))
        except Exception:
            return None
        return {
            "auc1": auc1,
            "auc2": auc2,
            "auc_diff": auc1 - auc2,
            "var": None,
            "z": None,
            "p_value": None,
        }

    # Normalize covariance to 2x2 to avoid shape/Index errors on degenerate cases
    cov = np.atleast_2d(cov)
    if cov.shape != (2, 2):
        full = np.zeros((2, 2))
        r, c = cov.shape
        full[:r, :c] = cov
        cov = full
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        z = np.inf
        p = 0.0
    else:
        z = diff / np.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    return {
        "auc1": float(aucs[0]),
        "auc2": float(aucs[1]),
        "auc_diff": diff,
        "var": var,
        "z": float(z),
        "p_value": float(p),
    }


def _compute_target_value_counts(
    fileobj,
    target_col: str,
    chunk_size: int = 100_000,
) -> Tuple[Optional[pd.Series], Optional[str], bool]:
    """
    Returns (series, error_message, missing_column_flag) for value counts of the target column.
    Chunked reads keep memory usage manageable on very large files.
    """
    if not fileobj:
        return None, "Missing file object.", False

    original_pos = None
    try:
        original_pos = fileobj.tell()
    except Exception:
        original_pos = None

    try:
        try:
            fileobj.seek(0)
        except Exception:
            pass

        header = pd.read_csv(fileobj, nrows=0)
        columns = header.columns.tolist()
        if target_col not in columns:
            return None, None, True

        try:
            fileobj.seek(0)
        except Exception:
            pass

        counts: Dict[Any, int] = {}
        for chunk in pd.read_csv(fileobj, usecols=[target_col], chunksize=chunk_size):
            vc = chunk[target_col].value_counts(dropna=False)
            for val, count in vc.items():
                counts[val] = counts.get(val, 0) + int(count)

        series = pd.Series(counts).sort_values(ascending=False) if counts else pd.Series(dtype="int64")
        return series, None, False
    except Exception as exc:
        return None, str(exc), False
    finally:
        if original_pos is not None:
            try:
                fileobj.seek(original_pos)
            except Exception:
                pass


def _prepare_features_for_smote_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical/text columns into numeric encodings so SMOTE can run.
    Mirrors the ColumnTransformer + OneHotEncoder used later, albeit simplified.
    """
    if df.empty:
        return df

    work = df.copy()

    cat_cols = work.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        # Include NaNs as their own column so we don't drop information
        work = pd.get_dummies(work, columns=cat_cols, dummy_na=True)

    bool_cols = work.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        work[bool_cols] = work[bool_cols].astype(int)

    non_numeric = work.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        work[non_numeric] = work[non_numeric].apply(pd.to_numeric, errors="coerce")

    work = work.fillna(0)
    return work


def _preview_smote_balance(
    fileobj,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Optional[pd.Series], Optional[str], bool]:
    """
    Mimics the training split + SMOTE application used later so we can show
    the balanced class counts that the models will actually see.
    """
    if not IMBLEARN_AVAILABLE:
        return None, "SMOTE preview requires 'imbalanced-learn'.", False

    original_pos = None
    try:
        original_pos = fileobj.tell()
    except Exception:
        original_pos = None

    try:
        try:
            fileobj.seek(0)
        except Exception:
            pass

        df = pd.read_csv(fileobj)

        if target_col not in df.columns:
            return None, None, True

        if df[target_col].nunique(dropna=False) < 2:
            return None, "Target column must contain at least two classes for SMOTE.", False

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = _prepare_features_for_smote_preview(X)
        if X.shape[1] == 0:
            return None, "No usable feature columns for SMOTE preview.", False

        try:
            X_train, _, y_train, _ = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError as exc:
            return None, str(exc), False

        try:
            from imblearn.over_sampling import SMOTE

            smoter = SMOTE(random_state=random_state)
            _, y_balanced = smoter.fit_resample(X_train, y_train)
        except Exception as exc:
            return None, str(exc), False

        counts = y_balanced.value_counts().sort_values(ascending=False)
        return counts, None, False
    except Exception as exc:
        return None, f"SMOTE preview failed: {exc}", False
    finally:
        if original_pos is not None:
            try:
                fileobj.seek(original_pos)
            except Exception:
                pass


@st.cache_data(show_spinner=False)
def _generate_profile_html(df: pd.DataFrame, title: str, minimal: bool) -> str:
    """
    Build a profiling report and return it as HTML.
    Caches by DataFrame content hash, title, and minimal flag.
    """
    if ProfileReport is None:
        raise ImportError(
            "Profiling library not found. Install 'ydata-profiling' (recommended) "
            "or 'pandas-profiling'."
        )

    kwargs = {"title": title}
    if minimal:
        kwargs["minimal"] = True

    try:
        profile = ProfileReport(df, **kwargs)
    except TypeError:
        # Older releases may not accept 'minimal' kwarg
        kwargs.pop("minimal", None)
        profile = ProfileReport(df, **kwargs)
        if minimal:
            # Try toggling via config when supported
            try:
                profile.config.set_option("minimal", True)
            except Exception:
                pass

    # Prefer richer layout when supported
    try:
        profile.config.set_option("explorative", True)
    except Exception:
        pass

    return profile.to_html()


def _bytesig_of_upload(fobj) -> str:
    """
    Compute a stable short hash signature of an uploaded file's content
    without destroying its read pointer.
    Used by Step 1.25 to detect if inputs changed.
    """
    try:
        pos = fobj.tell()
    except Exception:
        pos = None
    try:
        # If it's an UploadedFile, it may expose getvalue()
        if hasattr(fobj, "getvalue"):
            data = fobj.getvalue()
        else:
            data = fobj.read()
    finally:
        try:
            if pos is not None:
                fobj.seek(pos)
        except Exception:
            pass

    if not isinstance(data, (bytes, bytearray)):
        data = bytes(str(data), "utf-8")

    return hashlib.md5(data).hexdigest()


class ExperimentSetupApp:
    """
    A class to encapsulate the Streamlit experiment setup wizard.
    """
    
    def __init__(self):
        """
        Initialize the app and set the page title.
        """
        st.title("Elucidate")

        # Initialize ResultManager for centralized file operations
        if RESULT_MANAGER_AVAILABLE:
            self.result_mgr = get_result_manager("results")
        else:
            self.result_mgr = None

        # Initialize session state if it doesn't exist
        if 'uploaded_files_map' not in st.session_state:
            st.session_state.uploaded_files_map = {} # Stores the actual UploadedFile objects
        if 'selected_datasets' not in st.session_state:
            st.session_state.selected_datasets = [] # Stores just the names
        
        if 'use_smote' not in st.session_state:
            st.session_state.use_smote = False
            
        if 'target_column' not in st.session_state:
            st.session_state.target_column = "target"
        if 'selected_model_groups' not in st.session_state:
            st.session_state.selected_model_groups = []
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = []
        
        if 'results' not in st.session_state:
            st.session_state.results = {} 
        
        if 'benchmark_results_df' not in st.session_state:
            st.session_state.benchmark_results_df = None # Will store a DataFrame
        if 'benchmark_auc_comparison' not in st.session_state:
            st.session_state.benchmark_auc_comparison = None 

        if 'run_shap' not in st.session_state:
            st.session_state.run_shap = False
            
        # --- NEW: Cache for full DataFrames for Step 6 ---
        if 'full_dfs' not in st.session_state:
            st.session_state.full_dfs = {}

        if 'feature_selection' not in st.session_state:
            # { dataset_name: [list of selected feature columns] }
            st.session_state.feature_selection = {}

        if "fi_results_cache" not in st.session_state:
            # { dataset_name: payload }, same structure you already display (rf/lr/merged/meta)
            st.session_state.fi_results_cache = {}

        if "fi_signature" not in st.session_state:
            # tuple that identifies what the cache corresponds to (files+target)
            st.session_state.fi_signature = None

        # run once early in the app
        def ds_key(name):
            return name.replace('.csv', '').replace(' ', '_')
            
        if "benchmarks" in st.session_state:
            for k in list(st.session_state.benchmarks.keys()):
                nk = ds_key(k)
                if nk != k and nk not in st.session_state.benchmarks:
                    st.session_state.benchmarks[nk] = st.session_state.benchmarks.pop(k)

        if "fi_results_cache" not in st.session_state:
            st.session_state.fi_results_cache = {}
        if "fi_signature" not in st.session_state:
            st.session_state.fi_signature = None
        if "fi_stale" not in st.session_state:
            st.session_state.fi_stale = False


    def _on_preprocessing_change(self):
            """Resets results if preprocessing options change."""
            st.session_state.results = {}
            st.session_state.benchmark_results_df = None
            st.session_state.benchmark_auc_comparison = None
            st.session_state.run_shap = False 
            st.session_state.full_dfs = {} # --- NEW: Clear DF cache ---

            # --- NEW: also clear feature-importance cache/state ---
            st.session_state.fi_results_cache = {}
            st.session_state.fi_signature = None
            st.session_state.fi_stale = False
            
            # --- ADD THIS LINE ---
            st.session_state.pop("global_shap_dfs", None)


    def _render_step_1_dataset_selection(self):
        """
        Renders the UI for dataset selection (Step 1).
        """
        st.header("📁 Step 1: Upload & Select Datasets")
        
        with st.expander("Upload datasets in experiment", expanded=True):
            uploads = st.file_uploader(
                "Upload CSV files:", type="csv", accept_multiple_files=True,
                key="dataset_uploader"
            )

            # Handle file uploads/removals gracefully
            if uploads:
                new_file_names = [f.name for f in uploads]

                # Check if files have actually changed before clearing results
                current_datasets = st.session_state.get('selected_datasets', [])
                files_changed = set(new_file_names) != set(current_datasets)
                
                if files_changed:
                    # Files changed - clear results
                    try:
                        self._on_preprocessing_change()
                    except Exception as e:
                        st.error(f"Error clearing previous results: {e}")
                        # Continue anyway - don't block the update

                # Always store the file objects (even if same files, allows re-processing)
                st.session_state.uploaded_files_map = {f.name: f for f in uploads}
                # Store just the names for display and selection
                st.session_state.selected_datasets = new_file_names
                
                # Force re-display to ensure files are saved to results folder
                # Set a flag to indicate files need to be processed
                st.session_state['datasets_need_processing'] = True

            else:
                # No files uploaded (either initial state or all files removed)
                current_datasets = st.session_state.get('selected_datasets', [])
                if current_datasets:
                    # Files were removed - clear state
                    try:
                        self._on_preprocessing_change()
                    except Exception as e:
                        st.error(f"Error clearing results: {e}")

                st.session_state.uploaded_files_map = {}
                st.session_state.selected_datasets = []


    def _display_step_1_results(self):
        """
        Displays the results from Step 1 based on session state.
        """
        if st.session_state.get("selected_datasets"):
            st.success(f"Datasets selected: {', '.join(st.session_state.selected_datasets)}")
        else:
            st.info("No datasets selected.")

        # Show 5 sample rows for each uploaded dataset (if available)
        if st.session_state.uploaded_files_map:
            for name in st.session_state.selected_datasets:
                fileobj = st.session_state.uploaded_files_map.get(name)
                if not fileobj:
                    continue
                # Put sample + counts inside a collapsed per-dataset expander
                with st.expander(f"Preview: {name}", expanded=False):
                    try:
                        # Always rewind before each read
                        try: 
                            fileobj.seek(0)
                        except Exception:
                            pass

                        # ---- 0) Sample preview (first 5 rows)
                        df_head = pd.read_csv(fileobj, nrows=5)
                        st.markdown("**Sample (first 5 rows)**")
                        st.dataframe(df_head)

                        # Always save sample preview to results/eda/ folder (overwrite if exists)
                        if self.result_mgr:
                            safe_name = self.result_mgr.sanitize_name(name)
                            sample_path = self.result_mgr.eda_dir / f"{safe_name}_sample_preview.csv"
                            df_head.to_csv(sample_path, index=False)

                        # ---- 1) Robust SHAPE (rows, cols) without loading full file
                        # Read header for columns
                        try:
                            fileobj.seek(0)
                        except Exception:
                            pass
                        header = pd.read_csv(fileobj, nrows=0)
                        cols = header.columns.tolist()
                        n_cols = len(cols)

                        # Count rows via chunked pass on any one column
                        CHUNK = 200_000
                        try:
                            fileobj.seek(0)
                        except Exception:
                            pass
                        n_rows = 0
                        for chunk in pd.read_csv(fileobj, usecols=[cols[0]] if cols else None,
                                                chunksize=CHUNK):
                            n_rows += len(chunk)

                        st.markdown(f"**Shape:** ({n_rows:,}, {n_cols:,})")

                        # Always save dataset metadata to results/eda/ folder
                        if self.result_mgr:
                            safe_name = self.result_mgr.sanitize_name(name)
                            metadata = pd.DataFrame({
                                'metric': ['rows', 'columns'],
                                'value': [n_rows, n_cols]
                            })
                            metadata_path = self.result_mgr.eda_dir / f"{safe_name}_metadata.csv"
                            metadata.to_csv(metadata_path, index=False)

                        # ---- 2) Info-style table (dtype + non-null counts), computed in chunks
                        if cols:
                            # Accumulators
                            non_null = {c: 0 for c in cols}
                            dtypes_seen = None

                            try:
                                fileobj.seek(0)
                            except Exception:
                                pass
                            for chunk in pd.read_csv(fileobj, chunksize=CHUNK):
                                # dtypes from first chunk are good enough in practice
                                if dtypes_seen is None:
                                    dtypes_seen = chunk.dtypes
                                # accumulate non-null counts
                                nn = chunk.notna().sum()
                                for c in nn.index:
                                    non_null[c] += int(nn[c])

                            info_df = pd.DataFrame({
                                "column": cols,
                                "non_null": [non_null[c] for c in cols],
                                "nulls": [n_rows - non_null[c] for c in cols],
                                "%_non_null": [
                                    (non_null[c] / n_rows * 100.0) if n_rows else float("nan")
                                    for c in cols
                                ],
                                "dtype": [str(dtypes_seen.get(c, "object")) if dtypes_seen is not None else "unknown"
                                        for c in cols],
                            })
                            # nicer sorting: non-null desc, then name
                            info_df = info_df.sort_values(by=["non_null", "column"], ascending=[False, True], ignore_index=True)
                            st.markdown("**Info (concise):**")
                            st.dataframe(info_df)

                            # Always save info table and missing values summary to results/eda/
                            if self.result_mgr:
                                safe_name = self.result_mgr.sanitize_name(name)
                                # Save the column info table
                                info_path = self.result_mgr.eda_dir / f"{safe_name}_column_info.csv"
                                info_df.to_csv(info_path, index=False)

                                # Create and save missing values summary
                                missing_summary = info_df[['column', 'nulls', '%_non_null']].copy()
                                missing_summary.columns = ['column', 'missing_count', 'percent_non_null']
                                self.result_mgr.save_eda_summary(
                                    dataset_name=name,
                                    summary_stats=None,
                                    target_distribution=None,
                                    correlation_matrix=None,
                                    missing_summary=missing_summary
                                )
                                missing_path = self.result_mgr.eda_dir / f"{safe_name}_missing_values.csv"

                        # ---- 3) Describe (bounded sample for safety)
                        DESC_ROWS = 50_000  # adjust if you want more/less fidelity vs speed
                        try:
                            fileobj.seek(0)
                        except Exception:
                            pass
                        df_desc_sample = pd.read_csv(fileobj, nrows=DESC_ROWS)
                        describe_df = df_desc_sample.describe(include='number').round(2)
                        st.markdown(f"**Describe() on first {min(DESC_ROWS, n_rows):,} rows (numeric columns):**")
                        st.dataframe(describe_df)

                        # Save describe table to results/eda/ folder
                        try:
                            if self.result_mgr and not describe_df.empty:
                                safe_name = self.result_mgr.sanitize_name(name)
                                # Use save_eda_summary which is the proper method
                                self.result_mgr.save_eda_summary(
                                    dataset_name=name,
                                    summary_stats=describe_df,
                                    target_distribution=None,  # Will save later
                                    correlation_matrix=None,
                                    missing_summary=None
                                )
                                stats_path = self.result_mgr.eda_dir / f"{safe_name}_summary_stats.csv"
                                st.success(f"✓ Saved summary stats to: {stats_path}")
                            elif not self.result_mgr:
                                st.warning("ResultManager not initialized - summary stats not saved")
                        except Exception as e:
                            st.error(f"Failed to save summary stats: {e}")

                        # ---- 4) Optional: Pairplot (scatter matrix) for numeric columns (bounded)
                        try:
                            num_cols = df_desc_sample.select_dtypes(include=["number"]).columns.tolist()
                        except Exception:
                            num_cols = []

                        st.markdown("**Pairplot (scatter matrix)**")
                        if not num_cols:
                            st.info("No numeric columns available for pairplot.")
                        else:
                            # Suggest up to 8 columns by default
                            max_pair_cols = 8
                            suggested = num_cols[:max_pair_cols]
                            key_cols = f"pairplot_cols_{name}"
                            selected_pair_cols = st.multiselect(
                                "Choose numeric columns for pairplot (max 8):",
                                options=num_cols,
                                default=suggested,
                                key=key_cols,
                                help="Select a subset of numeric columns to visualize. Pairplot is limited to 8 columns for performance.",
                            )

                            if selected_pair_cols:
                                if len(selected_pair_cols) > max_pair_cols:
                                    st.warning(f"Please select at most {max_pair_cols} columns. Currently selected: {len(selected_pair_cols)}")
                                else:
                                    btn_key = f"btn_pairplot_{name}"
                                    if st.button("Show pairplot (sampled)", key=btn_key):
                                        with st.spinner("Rendering pairplot (this may take a few seconds)..."):
                                            try:
                                                # Try to read a bounded sample directly from the file for memory safety
                                                try:
                                                    fileobj.seek(0)
                                                except Exception:
                                                    pass
                                                max_rows_pp = 2000
                                                try:
                                                    df_pp = pd.read_csv(fileobj, usecols=selected_pair_cols, nrows=max_rows_pp)
                                                except Exception:
                                                    # Fallback to slicing the describe/sample frame
                                                    df_pp = df_desc_sample[selected_pair_cols].sample(min(len(df_desc_sample), max_rows_pp), random_state=0)

                                                if df_pp is None or df_pp.empty:
                                                    st.warning("No data available to plot.")
                                                else:
                                                    try:
                                                        import seaborn as sns
                                                        import matplotlib.pyplot as plt
                                                        pp = sns.pairplot(df_pp)
                                                        st.pyplot(pp.fig)

                                                        # Save pairplot to results/eda/ folder
                                                        try:
                                                            if self.result_mgr:
                                                                saved_path = self.result_mgr.save_eda_visualization(
                                                                    fig=pp.fig,
                                                                    dataset_name=name,
                                                                    plot_type="pairplot",
                                                                    feature_name=None
                                                                )
                                                                st.success(f"✓ Saved pairplot to: {saved_path}")
                                                            else:
                                                                st.warning("ResultManager not initialized - pairplot not saved")
                                                        except Exception as e:
                                                            st.error(f"Failed to save pairplot: {e}")

                                                        plt.close(pp.fig)
                                                    except Exception as e_pp:
                                                        st.error(f"Pairplot failed: {e_pp}")
                                            except Exception as e:
                                                st.error(f"Could not prepare pairplot: {e}")

                    except Exception as e:
                        st.warning(f"Could not produce preview for {name}: {e}")

                    target = st.session_state.get('target_column', 'target')
                    counts_series, err, missing = _compute_target_value_counts(fileobj, target)
                    if missing:
                        st.info(f"Target column '{target}' not found in this file.")
                    elif err:
                        st.warning(f"Could not compute value counts for {name}: {err}")
                    elif counts_series is not None:
                        st.markdown(f"**Value counts (`{target}`)**")
                        st.write(counts_series.to_frame(name="count"))

                        # Save target distribution to results/eda/ folder
                        try:
                            if self.result_mgr:
                                # Update the previously saved summary with target distribution
                                self.result_mgr.save_eda_summary(
                                    dataset_name=name,
                                    summary_stats=None,  # Already saved
                                    target_distribution=counts_series,
                                    correlation_matrix=None,
                                    missing_summary=None
                                )
                                safe_name = self.result_mgr.sanitize_name(name)
                                target_path = self.result_mgr.eda_dir / f"{safe_name}_target_distribution.csv"
                                st.success(f"✓ Saved target distribution to: {target_path}")
                            else:
                                st.warning("ResultManager not initialized - target distribution not saved")
                        except Exception as e:
                            st.error(f"Failed to save target distribution: {e}")


    def _render_step_1_5_preprocessing_options(self):
        """
        Renders the UI for preprocessing options (Step 1.5).
        """
        st.header("⚙️ Step 1.5: Preprocessing Options")
        
        # Disable checkbox if imblearn is not installed
        smote_disabled = not IMBLEARN_AVAILABLE
        
        st.session_state.use_smote = st.checkbox(
            "Apply SMOTE (Synthetic Minority Over-sampling TEchnique)",
            value=st.session_state.use_smote,
            on_change=self._on_preprocessing_change,
            disabled=smote_disabled,
            help="If checked, SMOTE will be applied to the *training data* to handle class imbalance before model fitting. Requires 'imbalanced-learn' to be installed."
        )
        
        if smote_disabled:
            st.warning("SMOTE is disabled because the 'imbalanced-learn' library was not found. Please install it to enable this feature.")


    def _render_step_1_4_feature_selector(self):
        """
        Step 1.4: Let the user choose independent variables per dataset.
        Default = all columns except the target.
        """
        st.header("🎯 Step 1.4: Select Features")

        if not st.session_state.selected_datasets or not st.session_state.uploaded_files_map:
            st.info("Upload datasets in Step 1 to choose features.")
            return

        target = st.session_state.get("target_column", "target")

        for name in st.session_state.selected_datasets:
            fileobj = st.session_state.uploaded_files_map.get(name)
            if not fileobj:
                continue

            with st.expander(f"Choose features for: {name}", expanded=False):
                # ----------------------------------------------------------
                # 1️⃣ Derive candidate features for this dataset
                # ----------------------------------------------------------
                import pandas as pd

                target = st.session_state.get("target_column", "target")
                fobj = st.session_state.uploaded_files_map.get(name)

                try:
                    fobj.seek(0)
                    header_df = pd.read_csv(fobj, nrows=0)
                    all_cols = list(header_df.columns)
                except Exception as e:
                    st.warning(f"Could not read columns for {name}: {e}")
                    try:
                        fobj.seek(0)
                        all_cols = list(pd.read_csv(fobj, nrows=100).columns)
                    except Exception as e2:
                        st.error(f"Fallback read failed: {e2}")
                        all_cols = []

                # Drop any empty or unnamed columns
                all_cols = [c for c in all_cols if not str(c).startswith("Unnamed:")]

                # Exclude target column if present
                if target in all_cols:
                    candidate_features = [c for c in all_cols if c != target]
                else:
                    candidate_features = all_cols[:]

                # Deduplicate cleanly
                seen = set()
                candidate_features = [c for c in candidate_features if not (c in seen or seen.add(c))]

                # ----------------------------------------------------------
                # 2️⃣ Stable multiselect (default = all selected)
                # ----------------------------------------------------------
                key = f"feature_select_{name}"
                store_key = "feature_selection"

                # Initialize top-level store if missing
                if store_key not in st.session_state:
                    st.session_state[store_key] = {}

                # Initialize this dataset’s widget only once
                if key not in st.session_state:
                    # Start with all columns selected by default
                    st.session_state[key] = candidate_features[:]
                    st.session_state[store_key][name] = st.session_state[key][:]

                # ---------- Quick-select from Feature Importance (if available) ----------
                fi_cache = st.session_state.get("fi_results_cache", {})
                fi_payload = fi_cache.get(name)

                if fi_payload:
                    src_choice = st.radio(
                        "Feature-importance source",
                        ["Merged (RF/L1-LR)", "RandomForest only", "L1-LR only"],
                        horizontal=True,
                        key=f"fi_src_{name}",
                        help="Use the ranking produced in Step 1.25."
                    )

                    topn_choice = st.selectbox(
                        "Quick-select top features",
                        ["—", "Top 5", "Top 10", "Top 15", "Top 20"],
                        key=f"fi_topn_{name}",
                        help="Applies to the multiselect below. Re-runs of Step 3 are required."
                    )

                    # Build ranked list according to source
                    try:
                        if src_choice.startswith("Merged"):
                            ranked = list(fi_payload["merged"]["feature"])
                        elif src_choice.startswith("RandomForest"):
                            # assume 'rf' table is already sorted by importance desc
                            ranked = list(fi_payload["rf"]["feature"])
                        else:  # L1-LR only
                            # assume 'lr' table has absolute-coef ranking
                            ranked = list(fi_payload["lr"]["feature"])
                    except Exception:
                        ranked = []

                    # Filter to columns actually present in this dataset and not the target
                    ranked = [c for c in ranked if c in candidate_features]

                    # If user picked a Top-N, apply it to the multiselect value and reset results
                    top_lookup = {"Top 5": 5, "Top 10": 10, "Top 15": 15, "Top 20": 20}
                    if topn_choice in top_lookup and ranked:
                        N = top_lookup[topn_choice]
                        topN = ranked[:N]

                        # write into the multiselect's session key (seeded below)
                        mk = f"feature_select_{name}"
                        st.session_state[mk] = topN[:]  # overwrite selection

                        # mirror into canonical store
                        st.session_state["feature_selection"][name] = topN[:]

                        # changing features must invalidate downstream results
                        _reset_experiment_state()
                        st.info(f"Applied {topn_choice} from {src_choice}. Step 3 results reset.")
                else:
                    st.caption("Compute Step 1.25 first to enable Top-N quick-select.")



                # “Select all” / “Clear” buttons
                c1, c2, _ = st.columns([1, 1, 6])
                with c1:
                    if st.button("Select all", key=f"selall_{name}"):
                        st.session_state[key] = candidate_features[:]
                with c2:
                    if st.button("Clear", key=f"clear_{name}"):
                        st.session_state[key] = []

                # Multiselect reads and writes directly to its stable key
                sel = st.multiselect(
                    "Select independent variables (used downstream):",
                    options=candidate_features,
                    key=key,
                    help="All columns selected by default. Use buttons above to change selections.",
                )

                # Mirror the value into canonical store
                st.session_state[store_key][name] = list(sel)


    def _display_step_1_5_results(self):
        """
        Displays the results from Step 1.5 based on session state.
        """
        smote_enabled = st.session_state.use_smote
        target = st.session_state.get("target_column", "target")

        if smote_enabled:
            st.info("SMOTE (Oversampling) is **Enabled**.")
        else:
            st.info("SMOTE (Oversampling) is **Disabled**.")

        datasets = st.session_state.get("selected_datasets", [])
        files_map = st.session_state.get("uploaded_files_map", {})
        if not datasets or not files_map:
            st.caption("Upload datasets in Step 1 to inspect the current target distribution.")
            return

        st.markdown(f"**Current value counts for `{target}`**")

        for name in datasets:
            fileobj = files_map.get(name)
            if not fileobj:
                st.warning(f"Uploaded file for '{name}' is unavailable.")
                continue

            counts_series, err, missing = _compute_target_value_counts(fileobj, target)
            st.markdown(f"Dataset: `{name}`")
            if missing:
                st.info(f"Target column '{target}' not found in this file.")
                continue
            if err:
                st.warning(f"Could not compute value counts for {name}: {err}")
                continue
            if counts_series is None:
                st.info("No rows available to summarize.")
                continue

            counts_df = (
                counts_series.to_frame(name="count")
                .rename_axis("value")
                .reset_index()
            )
            total = counts_df["count"].sum()
            if total > 0:
                counts_df["percent"] = (counts_df["count"] / total).map(lambda x: f"{x:.2%}")
            else:
                counts_df["percent"] = "-"

            st.caption("Raw dataset distribution")
            st.dataframe(counts_df, use_container_width=True)

            if smote_enabled and IMBLEARN_AVAILABLE:
                sm_counts, sm_err, _ = _preview_smote_balance(fileobj, target)
                if sm_err:
                    st.warning(f"SMOTE-balanced preview for {name} failed: {sm_err}")
                elif sm_counts is not None:
                    sm_df = (
                        sm_counts.to_frame(name="count")
                        .rename_axis("value")
                        .reset_index()
                    )
                    sm_total = sm_df["count"].sum()
                    if sm_total > 0:
                        sm_df["percent"] = (sm_df["count"] / sm_total).map(lambda x: f"{x:.2%}")
                    else:
                        sm_df["percent"] = "-"
                    st.caption("Training split after SMOTE (80/20 stratified)")
                    st.dataframe(sm_df, use_container_width=True)
                    
                    # Save SMOTE distribution for reporting
                    if self.result_mgr:
                        try:
                            self.result_mgr.save_smote_distribution(
                                dataset_name=name,
                                original_counts=counts_series,
                                smote_counts=sm_counts
                            )
                        except Exception as save_err:
                            pass  # Silent fail - don't interrupt user flow

        if smote_enabled and IMBLEARN_AVAILABLE:
            st.caption(
                "Raw counts use the entire dataset. SMOTE preview applies the same 80/20 stratified split and SMOTE(random_state=42) that Step 3 uses for training."
            )
        elif smote_enabled:
            st.caption("SMOTE preview unavailable because the 'imbalanced-learn' dependency is missing.")
        else:
            st.caption(
                "Counts reflect the raw dataset before splitting. Enable SMOTE to preview the balanced training distribution."
            )


    def _render_step_1_3_ydata_profiles(self):
        """
        Step 1.3: Generate YData (pandas) profiling reports for one or more datasets.
        Produces embedded previews and per-dataset HTML downloads.
        """
        st.header("📈 Step 1.3: Data Profiling Reports (Optional)")

        if ProfileReport is None:
            st.error(
                "Profiling library not available. Install `ydata-profiling` "
                "(preferred) or `pandas-profiling` to enable this step."
            )
            return

        if not st.session_state.selected_datasets or not st.session_state.uploaded_files_map:
            st.info("Upload datasets in Step 1 to enable profiling.")
            return

        # --- Controls ---
        # multiselect: choose which datasets to profile (default = all currently selected)
        ds_to_profile = st.multiselect(
            "Choose datasets to profile:",
            options=st.session_state.selected_datasets,
            default=st.session_state.selected_datasets,
            key="ydata_ds_select",
            help="You can profile multiple datasets at once."
        )

        st.session_state.ydata_minimal_mode = st.toggle(
            "Use minimal mode (faster on large files)",
            value=st.session_state.get("ydata_minimal_mode", False),
            key="ydata_minimal_mode_toggle",
        )

        c1, c2 = st.columns([1, 3])
        with c1:
            run_clicked = st.button("Generate profiling reports", type="primary", key="btn_ydata_profile")
        with c2:
            st.caption("Reports are built on the entire file. For very large CSVs, enable minimal mode.")

        # --- Build reports when requested ---
        if run_clicked and ds_to_profile:
            with st.spinner("Building profiling reports..."):
                for name in ds_to_profile:
                    fobj = st.session_state.uploaded_files_map.get(name)
                    if not fobj:
                        st.warning(f"File object for '{name}' not found; skipping.")
                        continue

                    # Always rewind before each read
                    try:
                        fobj.seek(0)
                    except Exception:
                        pass

                    try:
                        # Load the full DataFrame for the profiling run
                        df_full = pd.read_csv(fobj)
                    except Exception as exc:
                        st.error(f"Could not read '{name}' for profiling: {exc}")
                        continue

                    try:
                        html = _generate_profile_html(
                            df_full, title=f"{name} — Profile", minimal=st.session_state.ydata_minimal_mode
                        )
                        out_name = f"{Path(name).stem}.html"
                        st.session_state.ydata_profiles[name] = {"html": html, "filename": out_name}
                    except Exception as exc:
                        st.error(f"Failed to create profile for '{name}': {exc}")
                        continue

                    # Store in session for display & download
                    out_name = Path(name).with_suffix(".html").name
                    st.session_state.ydata_profiles[name] = {
                        "html": html,
                        "filename": out_name,
                    }

            if ds_to_profile:
                st.success("Profiling complete.")

        # --- Display any cached/built reports with download buttons ---
        if st.session_state.ydata_profiles:
            st.subheader("Profiles")
            for name in st.session_state.selected_datasets:
                prof = st.session_state.ydata_profiles.get(name)
                if not prof:
                    continue

                with st.expander(f"Profile: {name}", expanded=False):
                    # Download button
                    st.download_button(
                        "Download HTML report",
                        data=prof["html"].encode("utf-8"),
                        file_name=prof["filename"],
                        mime="text/html",
                        key=f"dl_{name}",
                    )
                    # Embedded preview
                    st.components.v1.html(prof["html"], height=600, scrolling=True)

            
    def _render_step_2_model_selection(self):
        """
        Renders the UI for model and target selection (Step 2).
        """
        st.header("🎯 Step 2: Select Target Variable & Models")
        
        # ---------- Stable “Select model groups to run” (seed once, never snap-back) ----------
        available_model_groups = list(MODELS.keys())
        group_key = "selected_model_groups"

        # Seed exactly once (default = all groups). Do NOT reseed when empty.
        if group_key not in st.session_state:
            st.session_state[group_key] = available_model_groups[:]

        # Buttons that modify only this state
        c1, c2, _ = st.columns([1, 1, 6])
        with c1:
            if st.button("Select all model groups", key="selall_model_groups"):
                st.session_state[group_key] = available_model_groups[:]
                _reset_experiment_state()   # changing selection resets Step 3
        with c2:
            if st.button("Clear model groups", key="clear_model_groups"):
                st.session_state[group_key] = []
                _reset_experiment_state()   # changing selection resets Step 3

        # Reset Step 3 WHENEVER user changes the multiselect value
        def _on_model_groups_change():
            _reset_experiment_state()

        selected_groups = st.multiselect(
            "Select model groups to run:",
            options=available_model_groups,
            key=group_key,
            on_change=_on_model_groups_change,   # <— the crucial line
            help="All groups are selected on first load. Any change resets the Run Experiment results."
        )

        # (Optional) flatten to individual models for downstream use
        flat_model_list = []
        for g in selected_groups:
            flat_model_list.extend(MODELS.get(g, {}).keys())
        st.session_state.selected_models = flat_model_list


    def _display_step_2_results(self):
        """
        Displays the results from Step 2 based on session state.
        """
        if not st.session_state.get("target_column"):
            st.warning("Please enter a target column name.")
        else:
            st.success(f"Target column: '{st.session_state.get('target_column')}'")
        
        if st.session_state.get("selected_models"):
            st.success(f"Models to run: {', '.join(st.session_state.get('selected_models'))}")
        else:
            st.info("No models selected.")


    def _render_step_3_run_experiment(self):
        """
        Renders the "Run Experiment" button ONLY if results do not exist.
        The button's logic is contained here.
        """
        st.header("🔬 Step 3: Run Experiments")
        
        # Only show the button if the experiment hasn't been run yet
        if not st.session_state.get("results"):
            if st.button("Run Experiment"):
                # Clear any old benchmark results
                self._on_preprocessing_change() # Use this to clear everything
                
                with st.spinner("Running models on all datasets... This may take a moment."):
                    try:
                        # Get the list of actual UploadedFile objects to run on
                        # files_to_run = [
                        #     st.session_state.uploaded_files_map[name] 
                        #     for name in st.session_state.selected_datasets
                        # ]

                        # Get the list of actual UploadedFile objects to run on
                        # files_to_run = [ st.session_state.uploaded_files_map[name] for name in st.session_state.selected_datasets ]

                        # NEW: build filtered, in-memory CSVs based on feature selection
                        filtered_files = []
                        target = st.session_state.get("target_column", "target")

                        for name in st.session_state.selected_datasets:
                            fileobj = st.session_state.uploaded_files_map[name]
                            try:
                                try: fileobj.seek(0)
                                except Exception: pass
                                df_full = pd.read_csv(fileobj)

                                selected_feats = st.session_state.feature_selection.get(name)
                                if selected_feats is None or len(selected_feats) == 0:
                                    # default to all non-target columns if user didn’t select
                                    selected_feats = [c for c in df_full.columns if c != target]

                                cols_to_keep = [c for c in selected_feats if c in df_full.columns]
                                # Ensure target is present if available
                                if target in df_full.columns:
                                    cols_to_keep = cols_to_keep + [target]

                                df_reduced = df_full[cols_to_keep].copy()

                                # Keep a copy for later steps (SHAP/local analysis)
                                if 'full_dfs' in st.session_state:
                                    st.session_state.full_dfs[name] = df_reduced

                                # Serialize to CSV in-memory
                                out_name = name                      # <- do not append "__selected"
                                filtered_files.append(_df_to_named_bytesio(df_reduced, out_name))

                            except Exception as e:
                                st.error(f"Failed to apply feature selection to {name}: {e}")

                        # Pass filtered_files instead of raw uploads
                        files_to_run = filtered_files



                        # Create the dictionary of selected model groups to pass
                        selected_groups_dict = {
                            group: MODELS[group] 
                            for group in st.session_state.selected_model_groups
                            if group in MODELS
                        }
                        
                        st.session_state.results = run_experiment(
                            files_to_run,
                            st.session_state.target_column,
                            selected_groups_dict,
                            st.session_state.use_smote 
                        )
                        
                        # Check for global errors (e.g., SMOTE library missing)
                        if "error" in st.session_state.results:
                            st.error(st.session_state.results["error"])
                            st.session_state.results = {} # Clear the error
                        else:
                            st.success("Experiment complete!")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"An error occurred during the experiment: {e}")
                        st.session_state.results = {} # Clear partial results on failure
        else:
            # Results exist, so don't show the "Run" button.
            pass


    def _display_step_3_results(self):
        """
        Displays the results from the experiment run, grouped by dataset and model group.
        """
        if not st.session_state.get("results"):
            return
            
        st.subheader("Experiment Results")

        for dataset_name, dataset_results in st.session_state.results.items():
            st.markdown(f"### Results for: `{dataset_name}`")
            
            # --- MODIFIED: Check for dataset-level error ---
            if dataset_results.get("error"):
                st.error(f"Error processing this dataset: {dataset_results['error']}")
                continue
            
            # --- MODIFIED: Get the 'metrics' dictionary ---
            metrics_data = dataset_results.get("metrics", {})

            if not metrics_data:
                st.warning("No results were generated for this dataset.")
                continue

            for group_name, group_results in metrics_data.items():
                st.markdown(f"#### Model Group: {group_name}")
                
                try:
                    df = pd.DataFrame.from_dict(group_results, orient="index")
                    
                    if "error" in df.columns and len(df.columns) == 1:
                        st.dataframe(df) # Show the error DataFrame
                    else:
                        st.dataframe(
                            df.style.format(
                                {
                                    "AUC": "{:.4f}",
                                    "PCC": "{:.4f}",
                                    "F1": "{:.4f}",
                                    "Recall": "{:.4f}",
                                    "BS": "{:.4f}",
                                    "KS": "{:.4f}",
                                    "PG": "{:.4f}",
                                    "H": "{:.4f}",
                                },
                                na_rep="Error" 
                            )
                        )
                except Exception as e:
                    st.error(f"Could not display results for {group_name}: {e}")
                    st.json(group_results) 


    def _calculate_benchmarks(self):
        """
        Calculates benchmark models and average AUC comparison tables.
        Populates session state with two DataFrames.
        """
        results = st.session_state.get("results", {})
        if not results:
            st.warning("No results found. Please run Step 3 first.")
            return

        # 1. Aggregate all AUC scores for each model
        model_scores: Dict[str, Dict[str, List[float]]] = {} # {group: {model: [auc1, auc2, ...]}}
        
        for dataset_name, dataset_results in results.items():
            # --- MODIFIED: Check for error and get metrics ---
            if dataset_results.get("error"):
                continue
            metrics_data = dataset_results.get("metrics", {})
            
            for group_name, group_results in metrics_data.items():
                if group_name not in model_scores:
                    model_scores[group_name] = {}
                
                for model_name, metrics in group_results.items():
                    if model_name not in model_scores[group_name]:
                        model_scores[group_name][model_name] = []
                    
                    if "AUC" in metrics and pd.notna(metrics["AUC"]): # Check for errors/NaN
                        model_scores[group_name][model_name].append(metrics["AUC"])

        # 2. Find best model AND build average AUC comparison tables
        benchmark_models: Dict[str, str] = {} # {group: 'best_model_name'}
        auc_comparison_tables: Dict[str, pd.DataFrame] = {} 
        
        for group_name, models in model_scores.items():
            avg_aucs = {}
            for model_name, auc_list in models.items():
                if auc_list: # Only consider models that ran successfully
                    avg_aucs[model_name] = np.mean(auc_list)
            
            if avg_aucs: 
                avg_auc_df = pd.DataFrame.from_dict(avg_aucs, orient='index', columns=['Average AUC'])
                avg_auc_df = avg_auc_df.sort_values(by='Average AUC', ascending=False)
                auc_comparison_tables[group_name] = avg_auc_df

                best_model = max(avg_aucs, key=avg_aucs.get)
                benchmark_models[group_name] = best_model
        
        st.session_state.benchmark_auc_comparison = auc_comparison_tables

        if not benchmark_models:
            st.error("Could not determine benchmark models. No successful runs found.")
            return

        # 3. Build the final benchmark summary table data
        final_table_data = []
        
        for dataset_name, dataset_results in results.items():
            # --- MODIFIED: Check for error and get metrics ---
            if dataset_results.get("error"):
                continue
            metrics_data = dataset_results.get("metrics", {})
            
            for group_name, best_model_name in benchmark_models.items():
                if group_name in metrics_data and best_model_name in metrics_data[group_name]:
                    metrics = metrics_data[group_name][best_model_name]
                    
                    if "error" not in metrics:
                        # Get SMOTE status from session state
                        smote_used = st.session_state.get('use_smote', False)
                        row = {
                            'Dataset': dataset_name,
                            'Model Group': group_name,
                            'Benchmark Model': best_model_name,
                            'SMOTE': 'Yes' if smote_used else 'No',
                            **metrics 
                        }
                        final_table_data.append(row)
        
        if not final_table_data:
            st.error("Failed to build benchmark table. No valid metrics found.")
            return
            
        # 4. Create and store the final summary DataFrame
        df = pd.DataFrame(final_table_data)
        all_cols = [
            'Dataset', 'Model Group', 'Benchmark Model', 'SMOTE',
            'AUC', 'PCC', 'F1', 'Recall', 'BS', 'KS', 'PG', 'H'
        ]
        final_cols = [col for col in all_cols if col in df.columns]
        st.session_state.benchmark_results_df = df[final_cols]
        # Persist benchmark results to disk for reproducibility
        try:
            out_dir = Path(__file__).parent / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Use a deterministic filename so repeated runs overwrite the file
            fname = "benchmark_results.csv"
            out_path = out_dir / fname
            st.session_state.benchmark_results_df.to_csv(out_path, index=False)
            # Store the saved CSV path in session state for easy access
            st.session_state.benchmark_results_csv = str(out_path)
            # Store SMOTE metadata
            st.session_state.benchmark_smote_used = st.session_state.get('use_smote', False)
        except Exception as e:
            st.warning(f"Could not save benchmark results to disk: {e}")
        
        # Save ALL model results (not just benchmarks) for comprehensive reporting
        try:
            all_model_data = []
            smote_used = st.session_state.get('use_smote', False)
            
            for dataset_name, dataset_results in results.items():
                if dataset_results.get("error"):
                    continue
                metrics_data = dataset_results.get("metrics", {})
                
                for group_name, group_results in metrics_data.items():
                    for model_name, metrics in group_results.items():
                        if "error" not in metrics:
                            row = {
                                'Dataset': dataset_name,
                                'Model Group': group_name,
                                'Model': model_name,
                                'SMOTE': 'Yes' if smote_used else 'No',
                                **metrics
                            }
                            all_model_data.append(row)
            
            if all_model_data:
                all_models_df = pd.DataFrame(all_model_data)
                # Ensure consistent column ordering
                priority_cols = ['Dataset', 'Model Group', 'Model', 'SMOTE', 
                                'AUC', 'PCC', 'F1', 'Recall', 'BS', 'KS', 'PG', 'H']
                final_cols = [col for col in priority_cols if col in all_models_df.columns]
                all_models_df = all_models_df[final_cols]
                
                # Save using result_manager
                self.result_mgr.save_all_model_results(all_models_df)
                st.session_state.all_model_results_df = all_models_df
        except Exception as e:
            st.warning(f"Could not save all model results: {e}")
        
        # Save model comparison charts for all metrics
        try:
            metrics_to_chart = ["AUC", "PCC", "F1", "Recall", "BS", "KS", "PG", "H"]
            comparison_figure_paths = []
            
            for metric in metrics_to_chart:
                if metric in df.columns:
                    fig_path = self._save_model_comparison_png(out_dir / "figures", metric)
                    if fig_path:
                        comparison_figure_paths.append(fig_path)
            
            if comparison_figure_paths:
                st.session_state.benchmark_comparison_figures = comparison_figure_paths
        except Exception as e:
            st.warning(f"Could not save comparison charts: {e}")


    def _render_step_4_benchmark_analysis(self):
        st.header("📊 Step 4: Benchmark Analysis")

        has_results = bool(st.session_state.get("results"))

        # Button only sets intent and clears old outputs
        clicked = st.button(
            "Find Benchmark Models",
            disabled=not has_results,
            key="btn_benchmark_models"
        )

        if clicked:
            st.session_state["benchmark_requested"] = True
            st.session_state["benchmark_results_df"] = None
            st.session_state["benchmark_auc_comparison"] = None
            st.session_state["run_shap"] = False

        # Compute ONLY if the user requested it
        if has_results and st.session_state.get("benchmark_requested"):
            with st.spinner("Calculating benchmark models..."):
                self._calculate_benchmarks()
            st.session_state["benchmark_requested"] = False  # consume the intent
            if st.session_state.get("benchmark_results_df") is not None:
                st.success("Benchmark analysis complete!")


    def _display_step_4_results(self):
        """
        Displays the final benchmark results.
        """
        bench_df = st.session_state.get("benchmark_results_df")

        # Auto-load from CSV if session state is empty but CSV exists
        if bench_df is None:
            benchmark_csv_path = Path(__file__).parent / "results" / "benchmark_results.csv"
            if benchmark_csv_path.exists():
                try:
                    bench_df = pd.read_csv(benchmark_csv_path)
                    st.session_state["benchmark_results_df"] = bench_df
                    st.info("📂 Auto-loaded benchmark results from saved CSV")
                except Exception:
                    pass

        if bench_df is not None:
            st.subheader("Benchmark Model Summary")
            st.markdown("This table shows the full performance metrics for *only* the best model from each group on each dataset.")
            df = bench_df
            # Calculate height based on number of rows (header ~38px + row ~35px each, with min/max bounds)
            num_rows = len(df)
            dynamic_height = min(max(38 + (num_rows * 35), 150), 800)
            st.dataframe(
                df.style.format(
                    {"AUC":"{:.4f}","PCC":"{:.4f}","F1":"{:.4f}","Recall":"{:.4f}",
                    "BS":"{:.4f}","KS":"{:.4f}","PG":"{:.4f}","H":"{:.4f}"},
                    na_rep="N/A"
                ),
                height=dynamic_height
            )
            st.markdown("---")

            # --- Model Comparison Charts Subsection ---
            st.subheader("Model Comparison Charts")
            st.markdown("Compare benchmark metrics across models and datasets.")

            df_long = _prepare_benchmark_long(df)
            if df_long.empty:
                st.info("No benchmark metrics available for charts.")
            else:
                # Use the canonical metric list requested by the user
                metrics = ["AUC", "PCC", "F1", "Recall", "BS", "KS", "PG", "H"]
                # Build charts only for metrics present in the long dataframe
                charts = {m: make_metric_chart(df_long, m) for m in metrics}

                # Display charts in rows of up to 3 charts per row for a responsive layout
                def _chunk(seq, n):
                    for i in range(0, len(seq), n):
                        yield seq[i : i + n]

                present_metrics = [m for m in metrics if not df_long[df_long["metric"] == m].empty]
                for chunk in _chunk(present_metrics, 3):
                    cols = st.columns(len(chunk))
                    for c, met in zip(cols, chunk):
                        with c:
                            st.altair_chart(charts.get(met), use_container_width=True)

            # Continue with existing stat tests
            self._render_step_4_stat_tests()
        else:
            st.info("Run benchmark analysis to see the final summary table here.")


    def _render_step_4_stat_tests(self):
        """
        Statistical tests on paired model outputs (per dataset).
        """
        if not SCIPY_AVAILABLE:
            st.warning("SciPy is not installed. Install `scipy` to run Wilcoxon, McNemar, and DeLong tests.")
            return

        all_results = st.session_state.get("results", {})
        if not all_results:
            st.info("Run experiments in Step 3 before running statistical tests.")
            return

        # Initialize comparison history in session state
        if "comparison_history" not in st.session_state:
            st.session_state.comparison_history = []

        # Dataset selection area (use a container instead of an expander to avoid nested expanders)
        dataset = None
        with st.container():
            st.markdown("**Paired Comparison: Select Dataset**")
            st.caption("Choose a dataset to perform paired statistical comparisons (Wilcoxon, McNemar, DeLong).")
            ds_names = list(all_results.keys())
            dataset = st.selectbox("Dataset for paired comparison:", ds_names, key="stat_tests_dataset")

        if not dataset:
            st.info("Select a dataset in the 'Paired Comparison: Select Dataset' area to enable tests.")
            return

        data_block = all_results.get(dataset, {})
        models_block = data_block.get("models", {})
        data_dict = data_block.get("data", {})
        X_test = data_dict.get("X_test"); y_test = data_dict.get("y_test")
        if X_test is None or y_test is None:
            st.info("Stored test data not available for this dataset.")
            return
        model_options = []
        for group, models in models_block.items():
            for name, model in models.items():
                if model is not None:
                    model_options.append(f"{group}::{name}")
        if len(model_options) < 2:
            st.info("Need at least two trained models to compare on this dataset.")
            return

        c1, c2 = st.columns(2)
        choice_a = c1.selectbox("Model A", model_options, key="stat_model_a")
        remaining = [m for m in model_options if m != choice_a]
        choice_b = c2.selectbox("Model B", remaining, key="stat_model_b")

        col_btn1, col_btn2 = st.columns([1, 1])
        run_test = col_btn1.button("Run statistical tests", key="btn_stat_tests")
        if len(st.session_state.comparison_history) > 0:
            if col_btn2.button("Clear comparison history", key="btn_clear_history"):
                st.session_state.comparison_history = []
                st.rerun()

        if run_test:
            try:
                grp_a, name_a = choice_a.split("::", 1)
                grp_b, name_b = choice_b.split("::", 1)
                model_a = models_block.get(grp_a, {}).get(name_a)
                model_b = models_block.get(grp_b, {}).get(name_b)
                if model_a is None or model_b is None:
                    st.error("Selected models could not be located.")
                    return

                proba_a = model_a.predict_proba(X_test)[:, 1]
                proba_b = model_b.predict_proba(X_test)[:, 1]
                pred_a = model_a.predict(X_test)
                pred_b = model_b.predict(X_test)
            except Exception as e:
                st.error(f"Failed to score selected models: {e}")
                return

            # Wilcoxon on per-sample absolute error
            wil_stat, wil_p, med_diff = _wilcoxon_abs_error_test(y_test, proba_a, proba_b)
            b, c, chi2, mc_p = _mcnemar_test(y_test, pred_a, pred_b)
            delong_res = delong_roc_test(y_test, proba_a, proba_b)

            # Save to disk via result_manager
            try:
                if self.result_mgr:
                    saved_path = self.result_mgr.save_paired_comparison(
                        dataset_name=dataset,
                        model_a_name=choice_a,
                        model_b_name=choice_b,
                        wilcoxon_results={
                            "statistic": wil_stat,
                            "p_value": wil_p,
                            "median_abs_error_diff": med_diff
                        },
                        mcnemar_results={
                            "b": b,
                            "c": c,
                            "chi2": chi2,
                            "p_value": mc_p
                        },
                        delong_results=delong_res
                    )
                else:
                    st.warning("⚠️ Result manager not available - comparison not saved to disk")
            except Exception as e:
                st.error(f"❌ Could not save comparison to disk: {e}")
                import traceback
                st.code(traceback.format_exc())

            # Store results in comparison history
            comparison_result = {
                "dataset": dataset,
                "model_a": choice_a,
                "model_b": choice_b,
                "wilcoxon": {
                    "statistic": wil_stat,
                    "p_value": wil_p,
                    "median_abs_error_diff": med_diff
                },
                "mcnemar": {
                    "b": b,
                    "c": c,
                    "chi2": chi2,
                    "p_value": mc_p
                },
                "delong": delong_res
            }
            st.session_state.comparison_history.append(comparison_result)
            st.success(f"✅ Comparison {len(st.session_state.comparison_history)} completed! Results added below.")

        # Display all comparison results
        if len(st.session_state.comparison_history) > 0:
            st.markdown("---")
            st.markdown(f"### 📊 Comparison Results ({len(st.session_state.comparison_history)} total)")
            
            for idx, result in enumerate(st.session_state.comparison_history, 1):
                st.markdown("---")
                st.markdown(f"#### Comparison {idx}: {result['model_a']} vs {result['model_b']}")
                st.markdown(f"**Dataset:** {result['dataset']}")
                st.markdown(f"**Model A:** {result['model_a']}")
                st.markdown(f"**Model B:** {result['model_b']}")
                
                st.markdown("##### Wilcoxon signed-rank (absolute error per sample)")
                if result['wilcoxon']['statistic'] is None:
                    st.info("Wilcoxon test could not be computed (check data length or SciPy availability).")
                else:
                    st.write({
                        "statistic": result['wilcoxon']['statistic'],
                        "p_value": result['wilcoxon']['p_value'],
                        "median_abs_error_diff (A-B)": result['wilcoxon']['median_abs_error_diff']
                    })

                    # Human-readable summary
                    med_diff = result['wilcoxon']['median_abs_error_diff']
                    wil_p = result['wilcoxon']['p_value']
                    if med_diff is not None and wil_p is not None:
                        direction = "Model A" if med_diff < 0 else ("Model B" if med_diff > 0 else "Both models")
                        verb = "had a lower" if med_diff != 0 else "had similar"
                        significance = "statistically significant" if wil_p < 0.05 else "not statistically significant"
                        st.markdown(
                            f"{direction} {verb} median absolute error than the other, and this difference was {significance} (p={wil_p:.4g})."
                        )

                st.markdown("##### McNemar's test (paired classification outcomes)")
                if result['mcnemar']['chi2'] is None or result['mcnemar']['p_value'] is None:
                    reason = "no discordant pairs (models agreed on every case)" if result['mcnemar']['b'] is not None and result['mcnemar']['c'] is not None and (result['mcnemar']['b'] + result['mcnemar']['c']) == 0 else "insufficient data or missing predictions"
                    st.info(f"McNemar's test not computed: {reason}.")
                else:
                    st.write({
                        "b (A correct, B wrong)": result['mcnemar']['b'],
                        "c (A wrong, B correct)": result['mcnemar']['c'],
                        "chi2": result['mcnemar']['chi2'],
                        "p_value": result['mcnemar']['p_value']
                    })
                    st.caption("McNemar compares disagreements; low p_value means one model is more accurate on the discordant cases.")

                st.markdown("##### DeLong test for AUC difference")
                if result['delong'] is None:
                    st.info("DeLong test could not be computed (need SciPy and binary labels with both classes).")
                else:
                    st.write(result['delong'])
                    st.caption("Positive auc_diff favors Model A. Lower p_value indicates a significant AUC gap.")
            
            st.info("💡 You can run another comparison by selecting different models above and clicking 'Run statistical tests' again.")
        else:
            st.info("No comparisons run yet. Select models and click 'Run statistical tests' to begin.")


    def _render_step_5_shap_analysis(self):
        """
        Renders the merged SHAP analysis with global plots and reliability checks.
        """
        st.header("🔍 Step 5: Global SHAP Analysis")
        
        if not SHAP_AVAILABLE:
            st.error("SHAP library not found. Please install it to run this analysis: `pip install shap matplotlib`")
            return

        st.markdown("Generate SHAP summary plots (global feature importance) for the best-performing **benchmark model** from each dataset.")
        # Dataset selection for SHAP (multi-select). Defaults to all benchmarked datasets if available.
        bench_df = st.session_state.get("benchmark_results_df")
        available_ds = []
        if bench_df is not None and not bench_df.empty:
            try:
                available_ds = list(pd.unique(bench_df["Dataset"]))
            except Exception:
                available_ds = list(st.session_state.get("selected_datasets", []))
        else:
            # Fall back to uploaded datasets list if benchmarks are not yet computed
            available_ds = list(st.session_state.get("selected_datasets", []))

        # Determine default selection (do NOT assign to session_state before widget creation)
        default_sel = st.session_state.get("shap_selected_datasets", available_ds)

        # Create the multiselect widget. Streamlit will populate `st.session_state["shap_selected_datasets"]`.
        st.multiselect(
            "Datasets to run Global SHAP for (multi-select):",
            options=available_ds,
            default=default_sel,
            key="shap_selected_datasets",
            help="Choose one or more datasets to generate Global SHAP plots for. Default = all available benchmark datasets.",
        )
        
        # --- Stable SHAP toggle and settings ---
        st.markdown("**⚙️ SHAP Computation Mode**")
        st.session_state.use_stable_shap = st.checkbox(
            "Use Stable SHAP (multi-trial with rank stability)",
            value=st.session_state.get("use_stable_shap", False),
            help="Runs multiple resampled trials with stratified sampling for more robust estimates and rank stability metrics."
        )
        
        if st.session_state.use_stable_shap:
            c1, c2, c3 = st.columns(3)
            st.session_state.stable_shap_trials = c1.number_input(
                "Trials",
                min_value=1,
                max_value=50,
                value=st.session_state.get("stable_shap_trials", 1),
                help="Number of resamples for rank stability. Use 1 for a quick single resample/stability check."
            )
            st.session_state.stable_shap_bg_size = c2.number_input(
                "Background size",
                min_value=50,
                max_value=2000,
                value=st.session_state.get("stable_shap_bg_size", 50),
                step=50,
                help="Background sample size per trial."
            )
            st.session_state.stable_shap_explain_size = c3.number_input(
                "Explain size",
                min_value=50,
                max_value=2000,
                value=st.session_state.get("stable_shap_explain_size", 50),
                step=50,
                help="Test sample size per trial."
            )
            st.caption(
                f"⏱️ Estimated time: ~{st.session_state.stable_shap_trials}× slower than single-shot SHAP. "
                "Provides rank stability metrics (avg_rank, std_rank) in the global table."
            )
        else:
            st.caption("Standard single-shot SHAP (fast, no rank stability metrics).")
        
        st.markdown("---")
        st.warning("This can be slow, especially for many datasets. Plots are *not* cached.")


        if st.button("Generate Global SHAP Plots"): # Renamed button
            st.session_state.run_shap = True
        
        # If requested to run SHAP now, generate fresh plots; otherwise,
        # if cached global SHAP results exist, display the cached summaries
        if st.session_state.run_shap:
            self._display_step_5_results()
        elif 'global_shap_dfs' in st.session_state and st.session_state.global_shap_dfs:
            # Display cached SHAP summaries so they persist across other actions
            self._display_cached_global_shap()



    def _display_step_5_results(self):
            """
            Retrieves models and data to generate SHAP plots in two columns.
            """
            benchmark_df = st.session_state.get("benchmark_results_df")
            all_results = st.session_state.get("results", {})

            if benchmark_df is None or not all_results:
                st.error("Benchmark results are missing. Cannot run SHAP.")
                return

            # Reduce to a single best benchmark per dataset (highest AUC) to save time
            try:
                best_per_dataset_idx = benchmark_df.groupby("Dataset")["AUC"].idxmax()
                reduced_benchmark_df = benchmark_df.loc[best_per_dataset_idx].reset_index(drop=True)
            except Exception:
                reduced_benchmark_df = benchmark_df

            # Ensure we clear the run flag when this function finishes so widget changes don't re-trigger SHAP runs
            try:
                with st.spinner("Generating Global SHAP plots... This may take several minutes."):
                    # Filter reduced benchmarks by user-selected datasets (if any)
                    selected_ds = st.session_state.get("shap_selected_datasets")
                    if selected_ds:
                        try:
                            reduced_benchmark_df = reduced_benchmark_df[reduced_benchmark_df["Dataset"].isin(selected_ds)].reset_index(drop=True)
                        except Exception:
                            # If filtering fails for any reason, fall back to the unfiltered set
                            pass
    
                    if reduced_benchmark_df.empty:
                        st.warning("No benchmark entries found for the selected datasets. Adjust your selection or run Step 4 first.")
                        return
    
                    for index, row in reduced_benchmark_df.iterrows():
                        dataset = row['Dataset']
                        group = row['Model Group']
                        model_name = row['Benchmark Model']
                        
                        st.subheader(f"Global SHAP Summary: `{dataset}` (Model: `{model_name}`)")
                        
                        try:
                            # Retrieve the stored model and data from the results dictionary
                            model_data = all_results.get(dataset, {})
                            model_to_explain = model_data.get('models', {}).get(group, {}).get(model_name)
                            X_train = model_data.get('data', {}).get('X_train')
                            X_test = model_data.get('data', {}).get('X_test')
                            y_train = model_data.get('data', {}).get('y_train')  # for stratified sampling
                            y_test = model_data.get('data', {}).get('y_test')    # for stratified sampling
                            
                            if model_to_explain is None or X_train is None or X_test is None:
                                st.warning(f"Could not find stored model or data for {dataset}. Skipping.")
                                continue
                            
                            # --- SHAP COMPUTATION ---
                            
                            # Choose stable or standard SHAP based on toggle
                            use_stable = st.session_state.get("use_stable_shap", False)
                            
                            if use_stable:
                                st.caption(f"🔄 Using Stable SHAP ({st.session_state.stable_shap_trials} trials)...")
                                sv, explain_data_sample_df, shap_global_df = get_shap_values_stable(
                                    model_to_explain,
                                    X_train,
                                    X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                    n_trials=int(st.session_state.stable_shap_trials),
                                    bg_size=int(st.session_state.stable_shap_bg_size),
                                    explain_size=int(st.session_state.stable_shap_explain_size),
                                )
                            else:
                                sv, explain_data_sample_df, shap_global_df = get_shap_values(
                                    model_to_explain, X_train, X_test
                                )
                            
                            # --- Cache the global SHAP df for Step 6 ---
                            if 'global_shap_dfs' not in st.session_state:
                                st.session_state.global_shap_dfs = {}
                            st.session_state.global_shap_dfs[dataset] = shap_global_df
                            
                            # --- Display rank stability metrics if using stable SHAP ---
                            if use_stable and "avg_rank" in shap_global_df.columns and "std_rank" in shap_global_df.columns:
                                st.markdown("##### 📊 Global SHAP Table (with Rank Stability)")
                                display_cols = ["rank", "feature", "abs_mean_shap", "mean_shap", "std_shap", "avg_rank", "std_rank"]
                                display_cols = [c for c in display_cols if c in shap_global_df.columns]
                                st.dataframe(
                                    shap_global_df[display_cols].head(20).style.format({
                                        "abs_mean_shap": "{:.4f}",
                                        "mean_shap": "{:.4f}",
                                        "std_shap": "{:.4f}",
                                        "avg_rank": "{:.2f}",
                                        "std_rank": "{:.2f}",
                                    }),
                                    use_container_width=True
                                )
                                st.caption(
                                    "**avg_rank**: average rank across trials (lower = more important). "
                                    "**std_rank**: rank stability (lower = more stable)."
                                )
                            # 2. Create two columns for the plots
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                
                                st.markdown("##### Summary Plot (Bar)")
                                st.caption("Average impact (magnitude) of each feature.")
                                fig, ax = plt.subplots()
                                # Bar: explicitly pass feature names (some SHAP versions need this)
                                shap_vals = sv.values if hasattr(sv, "values") else sv
                                # Robustly coerce features to numeric floats for plotting
                                plot_df = explain_data_sample_df.copy()
                                for c in plot_df.columns:
                                    # Try numeric coercion first
                                    coerced = pd.to_numeric(plot_df[c], errors='coerce')
                                    if coerced.notna().sum() >= max(1, int(0.1 * len(plot_df))):
                                        # If at least 10% of values parse as numeric, keep numeric coercion
                                        plot_df[c] = coerced
                                    else:
                                        # Else use categorical codes (preserves distinct categories)
                                        try:
                                            plot_df[c] = pd.Categorical(plot_df[c]).codes
                                        except Exception:
                                            plot_df[c] = pd.Series(pd.Categorical(plot_df[c]).codes, index=plot_df.index)
                                    # Fill remaining NaNs with column median (or 0 if median is NaN)
                                    try:
                                        med = pd.to_numeric(plot_df[c], errors='coerce').median()
                                        if pd.isna(med):
                                            med = 0.0
                                    except Exception:
                                        med = 0.0
                                    plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce').fillna(med)
    
                                # Ensure float numpy array
                                plot_array = plot_df.astype(float).values
    
                                # Ensure shap values are numeric numpy arrays (handle list/multiclass)
                                if isinstance(shap_vals, (list, tuple)):
                                    sv_list = []
                                    for part in shap_vals:
                                        part_arr = pd.DataFrame(np.asarray(part))
                                        part_arr = part_arr.apply(pd.to_numeric, errors='coerce').fillna(0)
                                        sv_list.append(part_arr.values.astype(float))
                                    shap_vals_arr = np.array(sv_list)
                                else:
                                    sv_df = pd.DataFrame(np.asarray(shap_vals))
                                    sv_df = sv_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                                    shap_vals_arr = sv_df.values.astype(float)
    
                                feat_names = list(shap_global_df["feature"]) if shap_global_df is not None and "feature" in shap_global_df.columns else list(explain_data_sample_df.columns)
                                shap.summary_plot(shap_vals_arr, plot_array,
                                                  plot_type="bar",
                                                  feature_names=feat_names,
                                                  show=False)
                                st.pyplot(fig)
                                try:
                                    out_fig_dir = Path(__file__).parent / "results" / "figures"
                                    out_fig_dir.mkdir(parents=True, exist_ok=True)
                                    safe_ds = str(dataset).replace(' ', '_').replace('.csv', '')
                                    safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                                    bar_path = out_fig_dir / f"shap_{safe_ds}_{safe_model}_bar.png"
                                    fig.savefig(bar_path, bbox_inches='tight', dpi=150)
                                    try:
                                        # Record the saved path in session state for stable lookup
                                        gmap = st.session_state.get("global_shap_figures", {})
                                        gmap.setdefault(safe_ds, {})["bar"] = str(bar_path)
                                        st.session_state["global_shap_figures"] = gmap
                                    except Exception:
                                        pass
                                except Exception as e_save_fig:
                                    try:
                                        st.warning(f"Could not save SHAP bar PNG: {e_save_fig}")
                                    except Exception:
                                        pass
                                plt.close(fig)
    
                            with col2:
                                st.markdown("##### Summary Plot (Dot)")
                                st.caption("Distribution of feature impacts (magnitude and direction).")
                                try:
                                    fig, ax = plt.subplots()
                                    shap_vals = sv.values if hasattr(sv, "values") else sv
    
                                    plot_df = explain_data_sample_df.copy()
                                    for c in plot_df.columns:
                                        try:
                                            if not pd.api.types.is_numeric_dtype(plot_df[c]):
                                                coerced = pd.to_numeric(plot_df[c], errors='coerce')
                                                if coerced.notna().sum() >= 1:
                                                    plot_df[c] = coerced
                                                else:
                                                    plot_df[c] = pd.Categorical(plot_df[c]).codes
                                        except Exception:
                                            plot_df[c] = pd.Categorical(plot_df[c]).codes
    
                                    try:
                                        plot_array = plot_df.astype(float).values
                                    except Exception:
                                        for c in plot_df.columns:
                                            plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce').fillna(0)
                                        plot_array = plot_df.astype(float).values
    
                                    feat_names = list(shap_global_df["feature"]) if shap_global_df is not None and "feature" in shap_global_df.columns else list(explain_data_sample_df.columns)
    
                                    # Coerce shap values to numeric arrays (try robustly) and attempt plotting.
                                    try:
                                        if isinstance(shap_vals, (list, tuple)):
                                            sv_list = []
                                            for part in shap_vals:
                                                part_arr = pd.DataFrame(np.asarray(part))
                                                part_arr = part_arr.apply(pd.to_numeric, errors='coerce').fillna(0)
                                                sv_list.append(part_arr.values.astype(float))
                                            shap_vals_arr = np.array(sv_list)
                                        else:
                                            sv_df = pd.DataFrame(np.asarray(shap_vals))
                                            sv_df = sv_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                                            shap_vals_arr = sv_df.values.astype(float)
    
                                        shap.summary_plot(shap_vals_arr, plot_array,
                                                          feature_names=feat_names,
                                                          show=False)
                                        st.pyplot(fig)
                                        try:
                                            out_fig_dir = Path(__file__).parent / "results" / "figures"
                                            out_fig_dir.mkdir(parents=True, exist_ok=True)
                                            safe_ds = str(dataset).replace(' ', '_').replace('.csv', '')
                                            safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                                            dot_path = out_fig_dir / f"shap_{safe_ds}_{safe_model}_dot.png"
                                            fig.savefig(dot_path, bbox_inches='tight', dpi=150)
                                            try:
                                                # Record the saved path in session state for stable lookup
                                                gmap = st.session_state.get("global_shap_figures", {})
                                                gmap.setdefault(safe_ds, {})["dot"] = str(dot_path)
                                                st.session_state["global_shap_figures"] = gmap
                                            except Exception:
                                                pass
                                        except Exception as e_save_fig2:
                                            try:
                                                st.warning(f"Could not save SHAP dot PNG: {e_save_fig2}")
                                            except Exception:
                                                pass
                                    except Exception as e_plot:
                                        st.error(f"SHAP dot-plot failed: {e_plot}")
                                        try:
                                            st.write({
                                                "shap_version": getattr(shap, "__version__", "unknown"),
                                                "shap_vals_type": str(type(shap_vals)),
                                                "shap_vals_asarray_dtype": str(np.asarray(shap_vals).dtype),
                                                "shap_vals_asarray_shape": str(np.asarray(shap_vals).shape),
                                                "plot_array_dtype": str(plot_array.dtype),
                                                "plot_array_shape": str(plot_array.shape),
                                            })
                                        except Exception:
                                            pass
    
                                        try:
                                            st.caption("Sample of plot data (first 5 rows)")
                                            st.write(pd.DataFrame(plot_array).head())
                                        except Exception:
                                            pass
                                        try:
                                            st.caption("Sample of shap values (as array, first 5 rows)")
                                            sv_sample = np.asarray(shap_vals)
                                            sv_show = sv_sample[0] if sv_sample.ndim == 3 else sv_sample
                                            st.write(pd.DataFrame(sv_show).head())
                                        except Exception:
                                            pass
                                    finally:
                                        plt.close(fig)
                                except Exception as e:
                                    st.error(f"Failed to generate dot plot: {e}")
                        
                        except Exception as e:
                            st.error(f"Failed to generate SHAP plot for {dataset} - {model_name}: {e}")
            finally:
                # Reset the run flag so subsequent widget changes do not cause automatic re-runs
                try:
                    st.session_state.run_shap = False
                except Exception:
                    pass

    def _display_cached_global_shap(self):
        """
        Display cached Global SHAP summaries from `st.session_state.global_shap_dfs`.
        This avoids recomputing SHAP and ensures the previously generated summaries
        remain visible even after other actions (like running reliability tests).
        """
        try:
            shap_cache = st.session_state.get("global_shap_dfs", {})
            if not shap_cache:
                st.info("No cached Global SHAP results available.")
                return

            bench_df = st.session_state.get("benchmark_results_df")
            for dataset, shap_df in shap_cache.items():
                # Try to determine the benchmark model name for display/context
                model_name = None
                try:
                    if bench_df is not None and "Dataset" in bench_df.columns:
                        row = bench_df[bench_df["Dataset"] == dataset]
                        if not row.empty:
                            model_name = row.iloc[0].get("Benchmark Model")
                except Exception:
                    model_name = None

                title = f"Global SHAP Summary: `{dataset}`"
                if model_name:
                    title += f" (Model: `{model_name}`)"
                st.subheader(title)

                try:
                    # Display rank-stability table when available
                    if shap_df is not None and "avg_rank" in shap_df.columns and "std_rank" in shap_df.columns:
                        st.markdown("##### 📊 Global SHAP Table (cached)")
                        display_cols = [c for c in ["rank", "feature", "abs_mean_shap", "mean_shap", "std_shap", "avg_rank", "std_rank"] if c in shap_df.columns]
                        st.dataframe(
                            shap_df[display_cols].head(20).style.format({
                                "abs_mean_shap": "{:.4f}",
                                "mean_shap": "{:.4f}",
                                "std_shap": "{:.4f}",
                                "avg_rank": "{:.2f}",
                                "std_rank": "{:.2f}",
                            }),
                            use_container_width=True,
                        )
                        st.caption(
                            "**avg_rank**: average rank across trials (lower = more important). "
                            "**std_rank**: rank stability (lower = more stable)."
                        )

                    # Try to display saved plot images if present
                    out_fig_dir = Path(__file__).parent / "results" / "figures"
                    safe_ds = str(dataset).replace(' ', '_').replace('.csv', '')
                    # Prefer explicit saved paths recorded in session state
                    bar_pattern = None
                    dot_pattern = None
                    try:
                        fig_map = st.session_state.get("global_shap_figures", {})
                        ds_map = fig_map.get(safe_ds, {}) if fig_map else {}
                        bar_path_saved = ds_map.get("bar")
                        dot_path_saved = ds_map.get("dot")
                        if bar_path_saved:
                            bar_pattern = Path(bar_path_saved)
                        if dot_path_saved:
                            dot_pattern = Path(dot_path_saved)
                    except Exception:
                        bar_pattern = dot_pattern = None

                    # If explicit mappings not available, fall back to previous inference logic
                    if (bar_pattern is None or not getattr(bar_pattern, 'exists', lambda: False)()) or (dot_pattern is None or not getattr(dot_pattern, 'exists', lambda: False)()):
                        # If model_name known, build deterministic filenames; else try globbing
                        if model_name:
                            safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                            candidate_bar = out_fig_dir / f"shap_{safe_ds}_{safe_model}_bar.png"
                            candidate_dot = out_fig_dir / f"shap_{safe_ds}_{safe_model}_dot.png"
                            if bar_pattern is None or not getattr(bar_pattern, 'exists', lambda: False)():
                                bar_pattern = candidate_bar if candidate_bar.exists() else bar_pattern
                            if dot_pattern is None or not getattr(dot_pattern, 'exists', lambda: False)():
                                dot_pattern = candidate_dot if candidate_dot.exists() else dot_pattern
                        else:
                            try:
                                candidates = list(out_fig_dir.glob(f"shap_{safe_ds}_*_bar.png"))
                                bar_pattern = candidates[0] if candidates else bar_pattern
                                candidates = list(out_fig_dir.glob(f"shap_{safe_ds}_*_dot.png"))
                                dot_pattern = candidates[0] if candidates else dot_pattern
                            except Exception:
                                pass

                    col1, col2 = st.columns(2)
                    with col1:
                        try:
                            if bar_pattern and bar_pattern.exists():
                                st.markdown("##### Summary Plot (Bar) - cached")
                                st.image(str(bar_pattern), use_container_width=True)
                            else:
                                # Create a small placeholder PNG so the UI shows a friendly message
                                out_fig_dir = Path(__file__).parent / "results" / "figures"
                                out_fig_dir.mkdir(parents=True, exist_ok=True)
                                placeholder_bar = out_fig_dir / f"shap_{safe_ds}_placeholder_bar.png"
                                if not placeholder_bar.exists():
                                    try:
                                        import matplotlib.pyplot as _plt
                                        figp, axp = _plt.subplots(figsize=(6, 3))
                                        axp.text(0.5, 0.5, 'No SHAP bar plot available\nRun Global SHAP to generate plots',
                                                 ha='center', va='center', fontsize=12)
                                        axp.axis('off')
                                        figp.tight_layout()
                                        figp.savefig(placeholder_bar, dpi=120, bbox_inches='tight')
                                        _plt.close(figp)
                                    except Exception:
                                        # If we cannot create a PNG, fall back to a caption
                                        st.caption("Bar plot not available. Run Global SHAP to generate plots.")
                                if placeholder_bar.exists():
                                    st.markdown("##### Summary Plot (Bar) - placeholder")
                                    st.image(str(placeholder_bar), use_container_width=True)
                        except Exception as e_disp:
                            st.warning(f"Could not render bar plot placeholder: {e_disp}")

                    with col2:
                        try:
                            if dot_pattern and dot_pattern.exists():
                                st.markdown("##### Summary Plot (Dot) - cached")
                                st.image(str(dot_pattern), use_container_width=True)
                            else:
                                out_fig_dir = Path(__file__).parent / "results" / "figures"
                                out_fig_dir.mkdir(parents=True, exist_ok=True)
                                placeholder_dot = out_fig_dir / f"shap_{safe_ds}_placeholder_dot.png"
                                if not placeholder_dot.exists():
                                    try:
                                        import matplotlib.pyplot as _plt
                                        figp, axp = _plt.subplots(figsize=(6, 3))
                                        axp.text(0.5, 0.5, 'No SHAP dot plot available\nRun Global SHAP to generate plots',
                                                 ha='center', va='center', fontsize=12)
                                        axp.axis('off')
                                        figp.tight_layout()
                                        figp.savefig(placeholder_dot, dpi=120, bbox_inches='tight')
                                        _plt.close(figp)
                                    except Exception:
                                        st.caption("Dot plot not available. Run Global SHAP to generate plots.")
                                if placeholder_dot.exists():
                                    st.markdown("##### Summary Plot (Dot) - placeholder")
                                    st.image(str(placeholder_dot), use_container_width=True)
                        except Exception as e_disp2:
                            st.warning(f"Could not render dot plot placeholder: {e_disp2}")

                except Exception as e:
                    st.warning(f"Could not display cached SHAP summary for {dataset}: {e}")
        except Exception as e_all:
            st.warning(f"Error while rendering cached Global SHAP: {e_all}")
    
    def _render_step_5_5_reliability_test(self):
        """
        Renders the UI for Step 6: Reliability tests for SHAP (rank stability + randomization sanity).
        Runs on-demand and persists results under `results/`.
        """
        st.header("✅ Step 6: SHAP Reliability Tests")

        if not SHAP_AVAILABLE:
            st.error("SHAP not available. Install `shap` to run reliability tests.")
            return

        # Preset speed options with on_change callback
        def _update_reliability_preset():
            """Update trials and background size when preset changes."""
            preset = st.session_state.rel_speed_preset
            if preset.startswith("Minimal"):
                st.session_state.rel_n_trials = 1
                st.session_state.rel_n_bg = 20
            elif preset.startswith("Quick"):
                st.session_state.rel_n_trials = 3
                st.session_state.rel_n_bg = 50
            elif preset.startswith("Thorough"):
                st.session_state.rel_n_trials = 30
                st.session_state.rel_n_bg = 500
            else:  # Balanced
                st.session_state.rel_n_trials = 10
                st.session_state.rel_n_bg = 200
        
        # Determine default preset based on current values
        # Default to Minimal for fastest execution
        current_trials = st.session_state.get("rel_n_trials", 1)
        current_bg = st.session_state.get("rel_n_bg", 20)
        
        if current_trials == 1 and current_bg == 20:
            default_preset = "Minimal (1 trial / 20 bg) - Fastest"
        elif current_trials == 3 and current_bg == 50:
            default_preset = "Quick (3 trials / 50 bg)"
        elif current_trials == 30 and current_bg == 500:
            default_preset = "Thorough (30 trials / 500 bg)"
        elif current_trials == 10 and current_bg == 200:
            default_preset = "Balanced (10 trials / 200 bg)"
        else:
            # Default to Minimal (minimum computational cost)
            default_preset = "Minimal (1 trial / 20 bg) - Fastest"
        
        preset = st.selectbox(
            "Speed preset",
            [
                "Minimal (1 trial / 20 bg) - Fastest",
                "Quick (3 trials / 50 bg)", 
                "Balanced (10 trials / 200 bg)", 
                "Thorough (30 trials / 500 bg)"
            ],
            index=[
                "Minimal (1 trial / 20 bg) - Fastest",
                "Quick (3 trials / 50 bg)", 
                "Balanced (10 trials / 200 bg)", 
                "Thorough (30 trials / 500 bg)"
            ].index(default_preset),
            key="rel_speed_preset",
            on_change=_update_reliability_preset,
            help="Minimal: Fastest, basic sanity check only. Quick: Good for testing. Balanced: Reliable results. Thorough: Publication-quality."
        )

        # Map presets to values for initial defaults
        if preset.startswith("Minimal"):
            default_trials, default_bg = 1, 20
        elif preset.startswith("Quick"):
            default_trials, default_bg = 3, 50
        elif preset.startswith("Thorough"):
            default_trials, default_bg = 30, 500
        else:
            default_trials, default_bg = 10, 200

        c1, c2 = st.columns(2)
        # Streamlit will populate `st.session_state` for the given keys.
        c1.number_input(
            "Trials (n)", min_value=1, max_value=200, value=int(st.session_state.get("rel_n_trials", default_trials)), key="rel_n_trials"
        )
        c2.number_input(
            "Background size", min_value=10, max_value=5000, step=10, value=int(st.session_state.get("rel_n_bg", default_bg)), key="rel_n_bg"
        )

        ds_options = list(st.session_state.get("selected_datasets", []))
        default_sel = st.session_state.get("reliability_selected_datasets", ds_options)
        st.multiselect(
            "Datasets to run reliability tests for:", options=ds_options, default=default_sel,
            key="reliability_selected_datasets",
            help="Choose datasets to run rank-stability and randomization sanity checks."
        )

        # If there are persisted reliability results, show a brief indicator
        persisted = st.session_state.get("reliability_results", {})
        if persisted:
            try:
                st.success(f"Persisted reliability results found for {len(persisted)} dataset(s).")
                ts_map = st.session_state.get("reliability_timestamps", {})
                for ds_name in persisted:
                    ts = ts_map.get(ds_name)
                    if ts:
                        st.markdown(f"- `{ds_name}` — last run: `{ts}`")
                    else:
                        st.markdown(f"- `{ds_name}` — last run: unknown")
                st.info("Use 'Run Reliability Tests' to refresh, or proceed to Step 6 to use cached results.")
            except Exception:
                # Non-fatal: skip rendering indicator if something goes wrong
                pass

        if st.button("Run Reliability Tests", key="btn_run_reliability"):
            # Capture current widget values before rerun
            # (Session state is already updated by the widgets above)
            st.session_state.run_reliability_test = True
            st.session_state.analysis_active = False  # <--- RESET Step 6 flag
            # Clear previous results
            st.session_state.reliability_results = {}
            st.session_state.reliability_ratios = {}
            st.session_state.reliability_texts = {}
            st.rerun()

        # Note: reliability computation is now triggered when analyzing a selected row
        # (Compute inline button removed to avoid premature heavy computations.)

        # If flagged to run OR there are existing persisted results, display them.
        # Previously the logic hid results when `run_reliability_test` was False
        # (which is set after a run). That caused previously-run outputs to
        # disappear when other actions (like Step 6) triggered a rerun. Show
        # results whenever `reliability_results` holds data, or when the user
        # explicitly requests a run.
        has_results = bool(st.session_state.get("reliability_results"))
        should_run = st.session_state.get("run_reliability_test", False)
        should_display = should_run or has_results

        if should_display:
            self._display_step_5_5_reliability_results(run_computation=should_run)


    def _display_step_5_5_reliability_results(self, run_computation: bool = False):
        """
        Runs the reliability computations and displays results.
        Saves CSV and TXT outputs deterministically under `results/`.
        
        Args:
            run_computation: If True, actually run the reliability tests. 
                           If False, only display existing results from session state or disk.
        """
        selected = st.session_state.get("reliability_selected_datasets", [])
        if not selected:
            st.info("Select one or more datasets above to run reliability tests.")
            return

        all_results = st.session_state.get("results", {})
        bench_df = st.session_state.get("benchmark_results_df")
        n_trials = int(st.session_state.get("rel_n_trials", 1))
        bg_size = int(st.session_state.get("rel_n_bg", 20))

        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        # If computation not requested, just load and display existing results
        if not run_computation:
            # First, try to display from session state
            session_results = st.session_state.get("reliability_results", {})
            session_texts = st.session_state.get("reliability_texts", {})
            
            if session_results:
                # Display results from session state (already computed in this session)
                for ds in selected:
                    safe_name = ds.replace(' ', '_').replace('.csv', '')
                    if safe_name in session_results:
                        st.markdown(f"#### Results for `{ds}`")
                        rank_df = session_results.get(safe_name)
                        summary_text = session_texts.get(safe_name, "")
                        sanity_ratio = st.session_state.get("reliability_ratios", {}).get(safe_name, float('nan'))
                        
                        if not pd.isna(sanity_ratio):
                            st.write({"sanity_ratio": sanity_ratio})
                        
                        if summary_text:
                            st.markdown("**Summary:**")
                            st.markdown(
                                f"<div style='font-size:16px; line-height:1.45; white-space:pre-wrap; max-width:100%;'>{summary_text}</div>",
                                unsafe_allow_html=True,
                            )
                        
                        if rank_df is not None and not rank_df.empty:
                            st.markdown("**Rank Stability Table:**")
                            st.dataframe(rank_df.head(20), use_container_width=True)
                return  # Don't load from disk if we have session state results
            
            # Otherwise, try loading from disk
            loaded_any = False
            for ds in selected:
                safe_name = ds.replace(' ', '_').replace('.csv', '')
                csv_path = out_dir / f"reliability_table_{safe_name}.csv"
                txt_path = out_dir / f"reliability_summary_{safe_name}.txt"
                if csv_path.exists() or txt_path.exists():
                    loaded_any = True
                    st.markdown(f"#### Loaded persisted results for `{ds}`")
                    try:
                        if csv_path.exists():
                            rank_df = pd.read_csv(csv_path)
                            st.session_state.reliability_results[safe_name] = rank_df
                            st.dataframe(rank_df.head(20), use_container_width=True)
                            st.caption(f"Loaded table from `{csv_path}`")
                        if txt_path.exists():
                            summary_text = txt_path.read_text(encoding='utf-8')
                            st.session_state.reliability_texts[safe_name] = summary_text
                            st.markdown(
                                f"<div style='font-size:16px; line-height:1.45; white-space:pre-wrap; max-width:100%;'>{summary_text}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Loaded summary from `{txt_path}`")
                    except Exception as e_load:
                        st.warning(f"Failed to load persisted reliability outputs for {ds}: {e_load}")
                else:
                    st.info(f"No persisted reliability outputs found for `{ds}`. Click 'Run Reliability Tests' to compute and save them.")

            # If we loaded any persisted outputs or displayed session results, do not recompute now.
            return

        # Otherwise proceed to compute (user requested a fresh run or no persisted outputs exist)
        for ds in selected:
            with st.spinner(f"Running reliability tests for {ds}..."):
                try:
                    data_block = all_results.get(ds, {})
                    data_dict = data_block.get("data", {})
                    models_block = data_block.get("models", {})

                    # Prefer the benchmark model if available
                    chosen_group = None
                    chosen_model_name = None
                    if bench_df is not None:
                        try:
                            row = bench_df[bench_df["Dataset"] == ds]
                            if not row.empty:
                                chosen_group = row.iloc[0]["Model Group"]
                                chosen_model_name = row.iloc[0]["Benchmark Model"]
                        except Exception:
                            chosen_group = None

                    # Fallback: pick the first available trained model
                    model_to_test = None
                    if chosen_group and chosen_model_name:
                        model_to_test = models_block.get(chosen_group, {}).get(chosen_model_name)

                    if model_to_test is None:
                        # pick any model present
                        for grp, grp_models in models_block.items():
                            for nm, mm in grp_models.items():
                                if mm is not None:
                                    model_to_test = mm
                                    chosen_group = grp
                                    chosen_model_name = nm
                                    break
                            if model_to_test is not None:
                                break

                    # Attempt to obtain train/test splits
                    X_train = data_dict.get("X_train")
                    X_test = data_dict.get("X_test")
                    y_train = data_dict.get("y_train")
                    y_test = data_dict.get("y_test")

                    if model_to_test is None:
                        st.warning(f"No trained model found for dataset {ds}. Skipping.")
                        continue

                    # If required data missing, try to reconstruct from uploaded full df
                    if X_train is None or X_test is None or y_train is None:
                        if ds in st.session_state.get("full_dfs", {}):
                            try:
                                full = st.session_state.full_dfs[ds]
                                target = st.session_state.get("target_column", "target")
                                if target in full.columns:
                                    X = full.drop(columns=[target])
                                    y = full[target]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
                                else:
                                    st.warning(f"Target column '{st.session_state.get('target_column')}' not found in {ds}. Skipping.")
                                    continue
                            except Exception as e:
                                st.warning(f"Failed to reconstruct train/test for {ds}: {e}")
                                continue
                        else:
                            st.warning(f"No stored data for {ds} (cannot run reliability tests). Skipping.")
                            continue

                    # Run rank stability and randomization sanity
                    try:
                        rank_df = shap_rank_stability(model_to_test, X_train, X_test, n_bg_samples=bg_size, n_trials=n_trials)
                    except Exception as e_rs:
                        st.error(f"Rank stability failed for {ds}: {e_rs}")
                        rank_df = None

                    try:
                        sanity_ratio = model_randomization_sanity(model_to_test, X_train, X_test, y_train, n_bg_samples=bg_size)
                    except Exception as e_ms:
                        st.error(f"Model randomization sanity check failed for {ds}: {e_ms}")
                        sanity_ratio = float('nan')

                    # Build textual summary
                    try:
                        summary_text = summarize_reliability(rank_df, sanity_ratio if sanity_ratio is not None else 0.0, n_trials, bg_size)
                    except Exception:
                        # Fallback summary
                        if rank_df is None or rank_df.empty:
                            summary_text = f"No rank-stability data for {ds}. Sanity ratio={sanity_ratio:.3f}"
                        else:
                            top = rank_df.sort_values('avg_rank').head(5)['feature'].tolist()
                            summary_text = f"Top features: {', '.join(map(str, top))}. Sanity ratio={sanity_ratio:.3f}."

                    # Persist outputs using ResultManager
                    safe_name = ds.replace(' ', '_').replace('.csv', '')
                    try:
                        if rank_df is not None and self.result_mgr:
                            # Use ResultManager to save to results/reliability/ subdirectory
                            saved_paths = self.result_mgr.save_reliability_results(
                                dataset_name=ds,
                                rank_df=rank_df,
                                sanity_ratio=sanity_ratio if sanity_ratio is not None else 0.0,
                                summary_text=summary_text
                            )
                            st.session_state.reliability_results[safe_name] = rank_df
                        elif rank_df is not None:
                            # Fallback if ResultManager not available
                            csv_path = out_dir / "reliability" / f"{safe_name}_rank.csv"
                            csv_path.parent.mkdir(parents=True, exist_ok=True)
                            rank_df.to_csv(csv_path, index=False)
                            st.session_state.reliability_results[safe_name] = rank_df
                            txt_path = out_dir / "reliability" / f"{safe_name}_summary.txt"
                            full_summary = f"Sanity Ratio: {sanity_ratio:.4f}\n\n{summary_text}"
                            txt_path.write_text(full_summary, encoding='utf-8')
                        st.session_state.reliability_texts[safe_name] = summary_text
                        st.session_state.reliability_ratios[safe_name] = float(sanity_ratio) if sanity_ratio is not None else float('nan')
                        # Record last-run timestamp for this dataset's reliability outputs
                        try:
                            st.session_state.reliability_timestamps[safe_name] = datetime.now().isoformat()
                        except Exception:
                            pass
                    except Exception as e_save:
                        st.warning(f"Could not persist reliability outputs for {ds}: {e_save}")

                    # Display
                    st.markdown(f"#### Results for `{ds}`")
                    st.write({"sanity_ratio": sanity_ratio})
                    st.markdown("**Summary:**")
                    # Use HTML-styled markdown to preserve wrapping and increase font size
                    try:
                        st.markdown(
                            f"<div style='font-size:16px; line-height:1.45; white-space:pre-wrap; max-width:100%;'>{summary_text}</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        # Fallback to plain text if HTML rendering is not allowed
                        st.text(summary_text)
                    if rank_df is not None:
                        st.markdown("**Rank-stability (top 20):**")
                        display_cols = [c for c in ["feature", "avg_rank", "std_rank"] if c in rank_df.columns]
                        st.dataframe(rank_df[display_cols].head(20).style.format({"avg_rank":"{:.2f}", "std_rank":"{:.2f}"}), use_container_width=True)

                except Exception as e_all:
                    st.error(f"Reliability pipeline failed for {ds}: {e_all}")

        # Done: reset the flag so UI changes do not auto-retrigger
        try:
            st.session_state.run_reliability_test = False
        except Exception:
            pass
    
    def _compute_reliability_for_selected(self):
        """
        Compute and persist reliability metrics for the currently-selected datasets.
        This runs synchronously and updates `st.session_state` and the `results/` folder.
        """
        selected = st.session_state.get("reliability_selected_datasets", [])
        if not selected:
            st.info("Select one or more datasets above to run reliability tests.")
            return

        all_results = st.session_state.get("results", {})
        bench_df = st.session_state.get("benchmark_results_df")
        n_trials = int(st.session_state.get("rel_n_trials", 1))
        bg_size = int(st.session_state.get("rel_n_bg", 20))

        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        for ds in selected:
            with st.spinner(f"Computing reliability metrics for {ds} (inline)..."):
                try:
                    data_block = all_results.get(ds, {})
                    data_dict = data_block.get("data", {})
                    models_block = data_block.get("models", {})

                    # Prefer the benchmark model if available
                    chosen_group = None
                    chosen_model_name = None
                    if bench_df is not None:
                        try:
                            row = bench_df[bench_df["Dataset"] == ds]
                            if not row.empty:
                                chosen_group = row.iloc[0]["Model Group"]
                                chosen_model_name = row.iloc[0]["Benchmark Model"]
                        except Exception:
                            chosen_group = None

                    # Fallback: pick the first available trained model
                    model_to_test = None
                    if chosen_group and chosen_model_name:
                        model_to_test = models_block.get(chosen_group, {}).get(chosen_model_name)

                    if model_to_test is None:
                        for grp, grp_models in models_block.items():
                            for nm, mm in grp_models.items():
                                if mm is not None:
                                    model_to_test = mm
                                    chosen_group = grp
                                    chosen_model_name = nm
                                    break
                            if model_to_test is not None:
                                break

                    # Attempt to obtain train/test splits
                    X_train = data_dict.get("X_train")
                    X_test = data_dict.get("X_test")
                    y_train = data_dict.get("y_train")
                    y_test = data_dict.get("y_test")

                    if model_to_test is None:
                        st.warning(f"No trained model found for dataset {ds}. Skipping.")
                        continue

                    # If required data missing, try to reconstruct from uploaded full df
                    if X_train is None or X_test is None or y_train is None:
                        if ds in st.session_state.get("full_dfs", {}):
                            try:
                                full = st.session_state.full_dfs[ds]
                                target = st.session_state.get("target_column", "target")
                                if target in full.columns:
                                    X = full.drop(columns=[target])
                                    y = full[target]
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
                                else:
                                    st.warning(f"Target column '{st.session_state.get('target_column')}' not found in {ds}. Skipping.")
                                    continue
                            except Exception as e:
                                st.warning(f"Failed to reconstruct train/test for {ds}: {e}")
                                continue
                        else:
                            st.warning(f"No stored data for {ds} (cannot run reliability tests). Skipping.")
                            continue

                    # Run rank stability and randomization sanity
                    try:
                        rank_df = shap_rank_stability(model_to_test, X_train, X_test, n_bg_samples=bg_size, n_trials=n_trials)
                    except Exception as e_rs:
                        st.error(f"Rank stability failed for {ds}: {e_rs}")
                        rank_df = None

                    try:
                        sanity_ratio = model_randomization_sanity(model_to_test, X_train, X_test, y_train, n_bg_samples=bg_size)
                    except Exception as e_ms:
                        st.error(f"Model randomization sanity check failed for {ds}: {e_ms}")
                        sanity_ratio = float('nan')

                    # Build textual summary
                    try:
                        summary_text = summarize_reliability(rank_df, sanity_ratio if sanity_ratio is not None else 0.0, n_trials, bg_size)
                    except Exception:
                        if rank_df is None or rank_df.empty:
                            summary_text = f"No rank-stability data for {ds}. Sanity ratio={sanity_ratio:.3f}"
                        else:
                            top = rank_df.sort_values('avg_rank').head(5)['feature'].tolist()
                            summary_text = f"Top features: {', '.join(map(str, top))}. Sanity ratio={sanity_ratio:.3f}."

                    # Persist outputs using ResultManager
                    safe_name = ds.replace(' ', '_').replace('.csv', '')
                    try:
                        if rank_df is not None and self.result_mgr:
                            # Use ResultManager to save to results/reliability/ subdirectory
                            saved_paths = self.result_mgr.save_reliability_results(
                                dataset_name=ds,
                                rank_df=rank_df,
                                sanity_ratio=sanity_ratio if sanity_ratio is not None else 0.0,
                                summary_text=summary_text
                            )
                            st.session_state.reliability_results[safe_name] = rank_df
                        elif rank_df is not None:
                            # Fallback if ResultManager not available
                            csv_path = out_dir / "reliability" / f"{safe_name}_rank.csv"
                            csv_path.parent.mkdir(parents=True, exist_ok=True)
                            rank_df.to_csv(csv_path, index=False)
                            st.session_state.reliability_results[safe_name] = rank_df
                            txt_path = out_dir / "reliability" / f"{safe_name}_summary.txt"
                            full_summary = f"Sanity Ratio: {sanity_ratio:.4f}\n\n{summary_text}"
                            txt_path.write_text(full_summary, encoding='utf-8')
                        st.session_state.reliability_texts[safe_name] = summary_text
                        st.session_state.reliability_ratios[safe_name] = float(sanity_ratio) if sanity_ratio is not None else float('nan')
                        try:
                            st.session_state.reliability_timestamps[safe_name] = datetime.now().isoformat()
                        except Exception:
                            pass
                    except Exception as e_save:
                        st.warning(f"Could not persist reliability outputs for {ds}: {e_save}")

                    # Display brief results
                    st.markdown(f"#### Results for `{ds}` (inline compute)")
                    st.write({"sanity_ratio": sanity_ratio})
                    st.markdown("**Summary:**")
                    try:
                        st.markdown(
                            f"<div style='font-size:16px; line-height:1.45; white-space:pre-wrap; max-width:100%;'>{summary_text}</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        st.text(summary_text)

                    if rank_df is not None:
                        st.markdown("**Rank-stability (top 20):**")
                        display_cols = [c for c in ["feature", "avg_rank", "std_rank"] if c in rank_df.columns]
                        st.dataframe(rank_df[display_cols].head(20).style.format({"avg_rank":"{:.2f}", "std_rank":"{:.2f}"}), use_container_width=True)

                except Exception as e_all:
                    st.error(f"Inline reliability pipeline failed for {ds}: {e_all}")

    def _compute_or_load_reliability_for_dataset(self, dataset, model, X_train, X_test, y_train, n_trials=None, bg_size=None):
        """
        Ensure reliability outputs exist for `dataset`. If persisted files are available,
        load them; otherwise compute rank stability and sanity, persist, and return.
        Returns (rank_df, sanity_ratio, summary_text)
        """
        n_trials = int(n_trials) if n_trials is not None else int(st.session_state.get("rel_n_trials", 1))
        bg_size = int(bg_size) if bg_size is not None else int(st.session_state.get("rel_n_bg", 20))

        safe_name = str(dataset).replace(' ', '_').replace('.csv', '')
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Update paths to use correct subdirectory structure
        csv_path = out_dir / "reliability" / f"{safe_name}_rank.csv"
        txt_path = out_dir / "reliability" / f"{safe_name}_summary.txt"

        # Try session_state first
        try:
            if safe_name in st.session_state.get("reliability_results", {}):
                rank_df = st.session_state.reliability_results[safe_name]
                sanity_ratio = st.session_state.reliability_ratios.get(safe_name)
                summary_text = st.session_state.reliability_texts.get(safe_name)
                return rank_df, sanity_ratio, summary_text
        except Exception:
            pass

        # Try loading persisted files from new location
        if csv_path.exists() or txt_path.exists():
            rank_df = None
            sanity_ratio = None
            summary_text = None
            try:
                if csv_path.exists():
                    rank_df = pd.read_csv(csv_path)
                    st.session_state.reliability_results[safe_name] = rank_df
                if txt_path.exists():
                    summary_content = txt_path.read_text(encoding='utf-8')
                    # Extract sanity ratio from first line if present
                    if summary_content.startswith("Sanity Ratio:"):
                        first_line = summary_content.split('\n')[0]
                        try:
                            sanity_ratio = float(first_line.split(':')[1].strip())
                        except:
                            pass
                        summary_text = '\n'.join(summary_content.split('\n')[2:])
                    else:
                        summary_text = summary_content
                    st.session_state.reliability_texts[safe_name] = summary_text
                    if sanity_ratio is not None:
                        st.session_state.reliability_ratios[safe_name] = sanity_ratio
                # Fallback: sanity_ratio might be embedded in session_state only
                if sanity_ratio is None:
                    sanity_ratio = st.session_state.reliability_ratios.get(safe_name)
            except Exception:
                pass
            return rank_df, sanity_ratio, summary_text

        # Otherwise compute (may be expensive) and persist
        rank_df = None
        sanity_ratio = None
        summary_text = None
        try:
            with st.spinner(f"Computing reliability for {dataset} (this may take time)..."):
                rank_df = shap_rank_stability(model, X_train, X_test, n_bg_samples=bg_size, n_trials=n_trials)
                sanity_ratio = model_randomization_sanity(model, X_train, X_test, y_train, n_bg_samples=bg_size)
                try:
                    summary_text = summarize_reliability(rank_df, float(sanity_ratio) if sanity_ratio is not None else 0.0, n_trials, bg_size)
                except Exception:
                    if rank_df is None or (hasattr(rank_df, 'empty') and rank_df.empty):
                        summary_text = f"No rank-stability data for {dataset}. Sanity ratio={sanity_ratio:.3f}"
                    else:
                        top = rank_df.sort_values('avg_rank').head(5)['feature'].tolist()
                        summary_text = f"Top features: {', '.join(map(str, top))}. Sanity ratio={sanity_ratio:.3f}."

                # Persist using ResultManager or fallback
                try:
                    if rank_df is not None and self.result_mgr:
                        # Use ResultManager to save to results/reliability/ subdirectory
                        saved_paths = self.result_mgr.save_reliability_results(
                            dataset_name=dataset,
                            rank_df=rank_df,
                            sanity_ratio=sanity_ratio if sanity_ratio is not None else 0.0,
                            summary_text=summary_text if summary_text is not None else ""
                        )
                        st.session_state.reliability_results[safe_name] = rank_df
                    elif rank_df is not None:
                        # Fallback if ResultManager not available
                        csv_path.parent.mkdir(parents=True, exist_ok=True)
                        rank_df.to_csv(csv_path, index=False)
                        st.session_state.reliability_results[safe_name] = rank_df
                        if summary_text is not None:
                            full_summary = f"Sanity Ratio: {sanity_ratio:.4f}\n\n{summary_text}"
                            txt_path.write_text(full_summary, encoding='utf-8')
                            st.session_state.reliability_texts[safe_name] = summary_text
                    st.session_state.reliability_ratios[safe_name] = float(sanity_ratio) if sanity_ratio is not None else float('nan')
                    st.session_state.reliability_timestamps[safe_name] = datetime.now().isoformat()
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"Failed to compute reliability for {dataset}: {e}")

        return rank_df, sanity_ratio, summary_text
    
    def _assign_reliability_bucket_unified(self, score, reference_scores=None):
        """
        Unified assignment of reliability buckets.

        If `reference_scores` (array-like) is provided, use quantile cutoffs
        (25/50/75%) computed from that array to assign labels:
            top quartile -> Reliable
            50-75%     -> Moderately Reliable
            25-50%     -> Questionable
            bottom 25% -> Unreliable

        Otherwise, fallback to sensible default thresholds to avoid
        everything falling into "Unreliable":
            >= 0.60 -> Reliable
            >= 0.40 -> Moderately Reliable
            >= 0.20 -> Questionable
            else   -> Unreliable
        """
        try:
            if pd.isna(score):
                return "Unknown"

            if reference_scores is not None:
                try:
                    # Allow pandas Series or list/ndarray
                    r = pd.Series(reference_scores).dropna()
                    if len(r) > 0:
                        q1, q2, q3 = r.quantile([0.25, 0.5, 0.75])
                        if score >= q3:
                            return "Reliable"
                        elif score >= q2:
                            return "Moderately Reliable"
                        elif score >= q1:
                            return "Questionable"
                        else:
                            return "Unreliable"
                except Exception:
                    # Fall back to defaults if quantile calc fails
                    pass

            # Default thresholds (fallback)
            if score >= 0.60:
                return "Reliable"
            elif score >= 0.40:
                return "Moderately Reliable"
            elif score >= 0.20:
                return "Questionable"
            else:
                return "Unreliable"
        except Exception:
            return "Unknown"

    def _compute_reliability_score_from_rankdf(self, rank_df, sanity_ratio, shap_values=None):
        """Compute per-row reliability score from the rank-stability DataFrame, sanity ratio, and local SHAP values.
        
        Args:
            rank_df: DataFrame with columns ['feature', 'avg_rank', 'std_rank']
            sanity_ratio: Scalar sanity ratio from model randomization test
            shap_values: Optional 1D numpy array of SHAP values for the current instance (same order as rank_df['feature'])
        
        Returns:
            dict with keys:
                - reliability_score: row-specific reliability if shap_values provided, else global
                - reliability_bucket: text bucket classification
                - er: global ER (mean of W_i)
                - w_signal: sanity signal
                - global_reliability: ER * w_signal (global baseline)
                - reliability_df: extended DataFrame with per-feature calculations
        """
        try:
            if rank_df is None or rank_df.empty:
                return {"reliability_score": float('nan'), "reliability_bucket": "Unknown", "er": float('nan'), 
                        "w_signal": float('nan'), "global_reliability": float('nan'), "reliability_df": None}
            
            df = rank_df.copy()
            if 'avg_rank' not in df.columns or 'std_rank' not in df.columns:
                return {"reliability_score": float('nan'), "reliability_bucket": "Unknown", "er": float('nan'), 
                        "w_signal": float('nan'), "global_reliability": float('nan'), "reliability_df": None}
            
            # Compute global feature weights W_i
            df['avg_rank'] = pd.to_numeric(df['avg_rank'], errors='coerce').fillna(0.0)
            df['std_rank'] = pd.to_numeric(df['std_rank'], errors='coerce').fillna(0.0)
            df['W_i'] = (1.0 / (1.0 + df['avg_rank'])) * (1.0 / (1.0 + df['std_rank']))
            
            # Compute global baseline
            er = float(df['W_i'].mean())
            try:
                w_signal = min(float(sanity_ratio), 3.0) / 3.0 if sanity_ratio is not None else 0.0
            except Exception:
                w_signal = 0.0
            
            global_reliability = er * w_signal
            global_reliability = max(0.0, min(1.0, float(global_reliability)))
            
            # If SHAP values provided, compute row-specific reliability
            if shap_values is not None:
                try:
                    shap_row = np.asarray(shap_values).flatten()
                    # Ensure length matches
                    if len(shap_row) == len(df):
                        df['shap_value'] = shap_row
                        df['abs_shap'] = np.abs(shap_row)
                        
                        # Compute local shares
                        denom = df['abs_shap'].sum()
                        if denom == 0:
                            df['local_share'] = 0.0
                        else:
                            df['local_share'] = df['abs_shap'] / denom
                        
                        # Compute local weighted reliability contribution
                        df['local_weighted_W'] = df['W_i'] * df['local_share']
                        
                        # Row-specific reliability score
                        row_reliability = w_signal * df['local_weighted_W'].sum()
                        row_reliability = max(0.0, min(1.0, float(row_reliability)))
                        reliability_score = row_reliability
                    else:
                        # Fallback to global if dimensions don't match
                        reliability_score = global_reliability
                except Exception:
                    # Fallback to global on any error
                    reliability_score = global_reliability
            else:
                # No SHAP values: use global reliability
                reliability_score = global_reliability
            
            # Classify into bucket
            if reliability_score >= 0.75:
                bucket = "Reliable"
            elif reliability_score >= 0.50:
                bucket = "Moderately Reliable"
            elif reliability_score >= 0.30:
                bucket = "Questionable"
            else:
                bucket = "Unreliable"
            
            return {
                "reliability_score": reliability_score,
                "reliability_bucket": bucket,
                "er": er,
                "w_signal": w_signal,
                "global_reliability": global_reliability,
                "reliability_df": df
            }
        except Exception:
            return {"reliability_score": float('nan'), "reliability_bucket": "Unknown", "er": float('nan'), 
                    "w_signal": float('nan'), "global_reliability": float('nan'), "reliability_df": None}
    

    # --- Step 6 Render Logic ---
    def _render_step_6_local_analysis(self):
            """
            Renders the UI for the new Step 6: Local SHAP Analysis.
            """

            # Auto-load saved analyses from disk if not already loaded
            if "local_shap_analyses" not in st.session_state or not st.session_state.local_shap_analyses:
                try:
                    import json
                    local_analyses_dir = Path(__file__).parent / "results" / "local_analyses"

                    if local_analyses_dir.exists():
                        json_files = list(local_analyses_dir.glob("local_analysis_*.json"))

                        if json_files:
                            loaded_analyses = {}
                            max_counter = 0

                            for json_file in json_files:
                                try:
                                    with open(json_file, 'r') as f:
                                        analysis_data = json.load(f)
                                        analysis_id = analysis_data.get("analysis_id", json_file.stem)
                                        loaded_analyses[analysis_id] = analysis_data

                                        # Extract counter from analysis_id (e.g., "local_analysis_5" -> 5)
                                        try:
                                            counter = int(analysis_id.split('_')[-1])
                                            max_counter = max(max_counter, counter)
                                        except (ValueError, IndexError):
                                            pass
                                except Exception:
                                    # Skip files that can't be loaded
                                    pass

                            # Update session state
                            if loaded_analyses:
                                st.session_state.local_shap_analyses = loaded_analyses
                                st.session_state.analysis_counter = max_counter
                except Exception:
                    # Non-fatal; continue even if auto-load fails
                    pass

                # Initialize empty dict and counter if still not present
                if "local_shap_analyses" not in st.session_state:
                    st.session_state.local_shap_analyses = {}
                if "analysis_counter" not in st.session_state:
                    st.session_state.analysis_counter = 0

            # with st.expander("Counterfactual constraints (optional)", expanded=False):
            #     immutable_str = st.text_input(
            #         "Immutable columns (comma-separated)", value="age,gender",
            #         help="These will never be changed in counterfactual search."
            #     )
            #     lb = st.text_area("Lower bounds JSON", value="{}", help='e.g. {"age": 18}')
            #     ub = st.text_area("Upper bounds JSON", value="{}", help='e.g. {"ltv": 1.0}')
            #
            #     # ✅ Store constraint UI selections so _display_step_6_results can use them
            #     st.session_state["immutable_str"] = immutable_str
            #     st.session_state["lb"] = lb
            #     st.session_state["ub"] = ub
            #     st.markdown(
            #         """
            #         **How to use these constraints**
            #         1. *Immutable columns*: comma-separated features that must never change, e.g. `age,gender`.
            #         2. *Lower bounds JSON*: enforce minimum values with JSON like `{"income": 30000, "ltv": 0.20}`.
            #         3. *Upper bounds JSON*: cap values via JSON such as `{"ltv": 0.80, "debt_to_income": 0.45}`.
            #         Leave any field empty/`{}` if you do not need that constraint.
            #         """
            #     )

            # Toggle for reliability features
            st.markdown("---")
            enable_reliability = st.checkbox(
                "🔬 Enable Row-Specific Reliability Analysis",
                value=False,
                key="enable_reliability_toggle",
                help="When enabled, Step 7 will compute row-specific reliability scores, show AI commentary with reliability context, and enable Step 7.5 for batch reliability computation."
            )
            if enable_reliability:
                st.info("✅ Reliability features enabled: Row-specific reliability scores, AI commentary, and Step 7.5 will be available.")
            else:
                st.info("ℹ️ Reliability features disabled: Only basic SHAP analysis will be shown.")

            st.header("🔎 Step 7: Local SHAP Analysis (Single Row Explanation)")

            if not SHAP_AVAILABLE or get_llm_explanation is None:
                st.error("SHAP or OpenAI libraries not found. Please install `shap`, `matplotlib`, `openai`, and `python-dotenv` to run this analysis.")
                return

            # --- NEW: Callback to reset analysis state ---
            def _reset_local_analysis_state():
                """Resets the flag that shows the analysis results."""
                if "analysis_active" in st.session_state:
                    st.session_state.analysis_active = False

            # 1. Select Dataset
            dataset_name = st.selectbox(
                "Select a dataset to analyze:",
                st.session_state.selected_datasets,
                index=0,
                key="local_analysis_dataset_select",
                on_change=_reset_local_analysis_state  # <--- ADDED CALLBACK
            )
            
            if not dataset_name:
                st.info("Upload a dataset in Step 1 to begin.")
                return

            try:
                # 2. Load and cache the full DataFrame
                if dataset_name not in st.session_state.full_dfs:
                    with st.spinner(f"Loading {dataset_name}..."):
                        fileobj = st.session_state.uploaded_files_map[dataset_name]
                        fileobj.seek(0)
                        st.session_state.full_dfs[dataset_name] = pd.read_csv(StringIO(fileobj.getvalue().decode("utf-8")))
                
                df = st.session_state.full_dfs[dataset_name]
                
                with st.expander("Show/Hide full data table"):
                    st.dataframe(df)
                
                # 3. Select Row
                max_idx = len(df) - 1
                row_index = st.number_input(
                    f"Select a row index (0 to {max_idx})",
                    min_value=0, max_value=max_idx, step=1,
                    key="local_analysis_row_select",
                    on_change=_reset_local_analysis_state  # <--- ADDED CALLBACK
                )
                
                # 4. Analyze Button
                if st.button("Analyze Selected Row"):
                    st.session_state.analysis_active = True  # <--- SET FLAG
                    # Don't touch run_reliability_test flag - let Step 5.5 results remain visible
                    st.rerun() # Force rerun to show results

                # --- NEW: Display results based on flag ---
                if st.session_state.get("analysis_active", False):
                    # Get the *current* values from state
                    current_dataset = st.session_state.local_analysis_dataset_select
                    current_row = st.session_state.local_analysis_row_select

                    if current_dataset in st.session_state.full_dfs:
                        current_df = st.session_state.full_dfs[current_dataset]
                        if current_row <= (len(current_df) - 1):
                            # Call display function based on the flag
                            self._display_step_6_results(current_dataset, current_df, current_row)
                        else:
                            st.error(f"Row index {current_row} is out of bounds for {current_dataset}. Max is {len(current_df) - 1}.")
                            st.session_state.analysis_active = False # Reset flag
                    else:
                        st.error(f"Data for {current_dataset} not found. Please reload.")
                        st.session_state.analysis_active = False # Reset flag

                # 5. View All Saved Analyses (moved after analysis display to show newly saved analyses)
                st.markdown("---")
                all_analyses = st.session_state.get("local_shap_analyses", {})
                if all_analyses:
                    with st.expander(f"📊 View All Saved Analyses ({len(all_analyses)} total)", expanded=False):
                        # Sort by analysis ID (chronological order)
                        sorted_ids = sorted(all_analyses.keys(), key=lambda x: int(x.split('_')[-1]))

                        for aid in sorted_ids:
                            rec = all_analyses[aid]
                            st.markdown(f"**{aid}**")
                            col1, col2, col3 = st.columns(3)
                            col1.write(f"Dataset: `{rec.get('dataset', 'N/A')}`")
                            col2.write(f"Row: `{rec.get('row_index', 'N/A')}`")
                            col3.write(f"Model: `{rec.get('model_name', 'N/A')}`")

                            col4, col5 = st.columns(2)
                            col4.write(f"Actual: `{rec.get('actual_target', 'N/A')}`")
                            col5.write(f"Predicted Prob: `{rec.get('predicted_prob', 0.0):.4f}`")

                            # Show reliability if available
                            rel_metrics = rec.get('reliability_metrics')
                            if rel_metrics:
                                st.markdown(f"**Reliability:** Score={rel_metrics.get('reliability_score', 0):.3f}, Classification={rel_metrics.get('reliability_bucket', 'N/A')}")

                            st.markdown("---")
                else:
                    st.info("ℹ️ No saved analyses yet. Analyze a row to start building your collection.")
                    
            except Exception as e:
                st.error(f"Failed to load or process dataset {dataset_name}: {e}")
                if "analysis_active" in st.session_state:
                    st.session_state.analysis_active = False # Reset on error

# --- NEW: Step 6 Display Logic ---
    def _display_step_6_results(self, dataset_name, df, row_index):
        """
        Displays the SHAP waterfall plot and LLM explanation for a single row.
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # Check if reliability is enabled
        enable_reliability = st.session_state.get("enable_reliability_toggle", False)
        
        benchmark_df = st.session_state.benchmark_results_df
        all_results = st.session_state.results
        target_col = st.session_state.target_column

        if benchmark_df is None:
             st.error("No benchmark models found. Please run Step 4 first.")
             return
             
        # Find the benchmark model for this specific dataset
        benchmark_row = benchmark_df[benchmark_df['Dataset'] == dataset_name]
        if benchmark_row.empty:
            st.error(f"No benchmark model found for {dataset_name}. Please run Step 4.")
            return
            
        # We take the first benchmark model found (in case of multiple groups)
        model_group = benchmark_row.iloc[0]['Model Group']
        model_name = benchmark_row.iloc[0]['Benchmark Model']
        
        st.info(f"Using benchmark model: **{model_name}** (from group: {model_group})")
        
        try:
            # Get the single row of data
            instance = df.iloc[[row_index]]
            instance_features = instance.drop(columns=[target_col])
            actual_target = instance[target_col].values[0]
            
            # Get the fitted model and training data
            model_data = all_results.get(dataset_name, {})
            model = model_data.get('models', {}).get(model_group, {}).get(model_name)
            X_train = model_data.get('data', {}).get('X_train')

            if model is None or X_train is None:
                st.error("Fitted model or training data not found. Please re-run Step 3.")
                return

            # --- Get Prediction and Explanation (in parallel) ---
            with st.spinner("Calculating local SHAP explanation..."):
                pred_proba = model.predict_proba(instance_features)[0]
                prob_class_1 = pred_proba[1] # Probability of class 1
                
                # Get the SHAP Explanation object
                explanation = get_local_shap_explanation(model, X_train, instance_features)
            
            # Display metrics
            st.subheader(f"Analysis for Row {row_index}")

            # Compute or load reliability BEFORE rendering the small metrics so we can show
            # a compact reliability indicator next to the predicted probability.
            try:
                n_trials = int(st.session_state.get("rel_n_trials", 1))
                bg_size = int(st.session_state.get("rel_n_bg", 20))
                model_block = all_results.get(dataset_name, {}).get('data', {})
                X_test = model_block.get('X_test')
                y_train = model_block.get('y_train')
                rank_df, sanity_ratio, summary_text = self._compute_or_load_reliability_for_dataset(
                    dataset_name, model, X_train, X_test, y_train, n_trials=n_trials, bg_size=bg_size
                )
                
                # Use the SAME unified function as batch computation
                threshold = float(st.session_state.get("classification_threshold", 0.5))
                try:
                    reliability_result = compute_single_row_reliability(
                        model=model,
                        X_train=X_train,
                        row_df=instance_features,
                        reliability_df=rank_df,
                        sanity_ratio=sanity_ratio,
                        threshold=threshold,
                        max_bg=200
                    )
                    # Convert to the format expected by the UI
                    # Prefer batch-derived reference distribution if available
                    reference_scores = None
                    try:
                        if "batch_reliability_results" in st.session_state and dataset_name in st.session_state.batch_reliability_results:
                            reference_scores = st.session_state.batch_reliability_results[dataset_name]["reliability_score"]
                    except Exception:
                        reference_scores = None

                    reliability_metrics = {
                        'reliability_score': reliability_result['reliability_score'],
                        'reliability_bucket': self._assign_reliability_bucket_unified(reliability_result['reliability_score'], reference_scores=reference_scores),
                        'er': reliability_result['ER_global'],
                        'w_signal': reliability_result['W_signal'],
                        'global_reliability': reliability_result['global_reliability'],
                        'reliability_df': rank_df  # Keep the rank_df for display
                    }
                    reliability_df = rank_df
                except Exception as e_rel:
                    st.warning(f"Could not compute unified reliability: {e_rel}")
                    reliability_metrics = None
                    reliability_df = None
            except Exception:
                rank_df = None
                sanity_ratio = None
                summary_text = None
                reliability_metrics = None
                reliability_df = None

            col1, col2, col3 = st.columns([1, 1, 1])
            col1.metric("Actual Target", f"{actual_target}")
            col2.metric("Predicted Probability (for Class 1)", f"{prob_class_1:.4f}")
            
            # Show reliability score only if enabled
            if enable_reliability:
                try:
                    if reliability_metrics is not None:
                        score = reliability_metrics.get('reliability_score')
                        bucket = reliability_metrics.get('reliability_bucket')
                        col3.metric("Reliability Score", f"{float(score):.3f}" if score is not None and not pd.isna(score) else "N/A")
                        # Show bucket as small caption below the metric
                        col3.markdown(f"**{bucket}**" if bucket else "")
                    else:
                        col3.metric("Reliability Score", "N/A")
                except Exception:
                    col3.metric("Reliability Score", "N/A")
            else:
                col3.markdown("")  # Empty space to maintain layout

            # Show detailed reliability metrics only if enabled
            if enable_reliability:
                st.markdown("---")
                try:
                    score = reliability_metrics.get('reliability_score') if reliability_metrics is not None else None
                    er = reliability_metrics.get('er') if reliability_metrics is not None else None
                    w_signal = reliability_metrics.get('w_signal') if reliability_metrics is not None else None
                    global_rel = reliability_metrics.get('global_reliability') if reliability_metrics is not None else None
                    
                    calc_method = (
                        "Row-specific: R(x) = W_signal * sum(W_i * local_share_i), where local_share_i = |SHAP_i| / sum(|SHAP|).\n"
                        "Global: ER = mean(W_i), W_i = (1/(1+avg_rank)) * (1/(1+std_rank)); Global_R = ER * W_signal."
                    )

                    st.markdown("**Row-Specific Reliability:**")
                    metrics_df = pd.DataFrame([
                        {"Metric": "Row Reliability Score", "Value": f"{float(score):.3f}" if score is not None and not pd.isna(score) else "N/A"},
                        {"Metric": "Calculation Method", "Value": calc_method},
                    ])
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    st.markdown("**Global Diagnostics:**")
                    global_df = pd.DataFrame([
                        {"Metric": "Global Reliability (ER * W_signal)", "Value": f"{float(global_rel):.3f}" if global_rel is not None and not pd.isna(global_rel) else "N/A"},
                        {"Metric": "ER (mean W_i)", "Value": f"{float(er):.4f}" if er is not None and not pd.isna(er) else "N/A"},
                        {"Metric": "W_signal (sanity scaled)", "Value": f"{float(w_signal):.4f}" if w_signal is not None and not pd.isna(w_signal) else "N/A"},
                        {"Metric": "Sanity Ratio", "Value": f"{float(sanity_ratio):.4f}" if sanity_ratio is not None and not pd.isna(sanity_ratio) else "N/A"},
                    ])
                    st.dataframe(global_df, use_container_width=True)
                except Exception as e_tab:
                    st.warning(f"Could not render reliability metrics table: {e_tab}")

                try:
                    # Bucket ranges as a DataFrame with highlighted selection
                    # Prefer to show batch-derived quantile buckets if available
                    if "batch_reliability_results" in st.session_state and dataset_name in st.session_state.batch_reliability_results:
                        try:
                            r_ref = st.session_state.batch_reliability_results[dataset_name]["reliability_score"]
                            q1, q2, q3 = pd.Series(r_ref).quantile([0.25, 0.5, 0.75])
                            buckets = [
                                {"Bucket": "Unreliable", "Min": float(pd.Series(r_ref).min()), "Max": float(q1)},
                                {"Bucket": "Questionable", "Min": float(q1), "Max": float(q2)},
                                {"Bucket": "Moderately Reliable", "Min": float(q2), "Max": float(q3)},
                                {"Bucket": "Reliable", "Min": float(q3), "Max": float(pd.Series(r_ref).max())},
                            ]
                        except Exception:
                            # Fall back to defaults below if any issue
                            buckets = None
                    else:
                        buckets = None

                    if buckets is None:
                        # Default, more lenient thresholds to avoid everything being 'Unreliable'
                        buckets = [
                            {"Bucket": "Reliable", "Min": 0.60, "Max": 1.0},
                            {"Bucket": "Moderately Reliable", "Min": 0.40, "Max": 0.60},
                            {"Bucket": "Questionable", "Min": 0.20, "Max": 0.40},
                            {"Bucket": "Unreliable", "Min": 0.0, "Max": 0.20},
                        ]

                    bdf = pd.DataFrame(buckets)
                    sel = reliability_metrics.get('reliability_bucket')
                    bdf['Selected'] = bdf['Bucket'] == sel
                    # Create a styler to highlight the selected row
                    def _highlight_selected(row):
                        return ['background-color: #d4edda' if row.Selected else '' for _ in row]
                    try:
                        styled = bdf.style.format({"Min": "{:.2f}", "Max": "{:.2f}"}).apply(_highlight_selected, axis=1)
                        st.markdown("**Reliability Bucket Ranges:**")
                        st.dataframe(styled, use_container_width=True)
                    except Exception:
                        # Fallback: show simple text list
                        st.markdown("**Reliability Bucket Ranges:**")
                        for _, r in bdf.iterrows():
                            mark = " ✔️" if r['Selected'] else ""
                            st.markdown(f"- {r['Bucket']} (range {r['Min']:.2f}-{r['Max']:.2f}){mark}")
                except Exception as e_buckets:
                    st.warning(f"Could not render bucket ranges: {e_buckets}")

                    # Optionally show top features reliability table and detailed calculation
                    if rank_df is not None and not getattr(rank_df, 'empty', True):
                        try:
                            st.markdown("**Top features (stability):**")
                            display_cols = [c for c in ["feature", "avg_rank", "std_rank"] if c in rank_df.columns]
                            df_top = rank_df[display_cols].head(10).copy()
                            # Compute per-feature weights for display
                            try:
                                df_top['avg_rank'] = df_top['avg_rank'].astype(float)
                                df_top['std_rank'] = df_top['std_rank'].astype(float)
                                df_top['W_rank'] = 1.0 / (1.0 + df_top['avg_rank'])
                                df_top['W_stab'] = 1.0 / (1.0 + df_top['std_rank'])
                                df_top['W'] = df_top['W_rank'] * df_top['W_stab']
                            except Exception:
                                pass

                            st.dataframe(df_top.style.format({"avg_rank":"{:.2f}", "std_rank":"{:.2f}", "W_rank":"{:.4f}", "W_stab":"{:.4f}", "W":"{:.4f}"}), use_container_width=True)

                            # Show detailed numeric calculation for this dataset
                            try:
                                # compute ER from full rank_df if available
                                tmp = rank_df.copy()
                                tmp['avg_rank'] = tmp['avg_rank'].astype(float)
                                tmp['std_rank'] = tmp['std_rank'].astype(float)
                                tmp['W_rank'] = 1.0 / (1.0 + tmp['avg_rank'])
                                df_top['W_stab'] = 1.0 / (1.0 + tmp['std_rank'])
                                tmp['W'] = tmp['W_rank'] * tmp['W_stab']
                                ER = float(tmp['W'].mean())
                            except Exception:
                                ER = None

                            try:
                                W_signal = min(float(sanity_ratio), 3.0) / 3.0 if sanity_ratio is not None and not pd.isna(sanity_ratio) else None
                            except Exception:
                                W_signal = None

                            # final score (use previously computed if available)
                            final_score = reliability_metrics.get('reliability_score') if reliability_metrics is not None else None
                            bucket = reliability_metrics.get('reliability_bucket') if reliability_metrics is not None else None

                            # Build a full-feature calculation table: W_rank, W_stab, W_i, and dataset-level ER/W_signal/RelScore
                            try:
                                calc_df = tmp[['feature', 'avg_rank', 'std_rank', 'W_rank', 'W_stab', 'W']].copy()
                                calc_df = calc_df.rename(columns={'W': 'W_i', 'avg_rank': 'avg_rank', 'std_rank': 'std_rank'})
                                # Add dataset-level scalars repeated per row for display
                                calc_df['ER'] = ER if ER is not None else float('nan')
                                calc_df['W_signal'] = W_signal if W_signal is not None else float('nan')
                                try:
                                    calc_df['RelScore'] = calc_df['ER'] * calc_df['W_signal']
                                except Exception:
                                    calc_df['RelScore'] = float('nan')

                                # Reorder columns for clarity
                                display_order = ['feature', 'avg_rank', 'std_rank', 'W_rank', 'W_stab', 'W_i', 'ER', 'W_signal', 'RelScore']
                                calc_df = calc_df[[c for c in display_order if c in calc_df.columns]]

                                st.markdown("**Reliability Calculation (per-feature table):**")
                                st.dataframe(calc_df.style.format({
                                    'avg_rank': '{:.2f}', 'std_rank': '{:.2f}', 'W_rank': '{:.4f}', 'W_stab': '{:.4f}', 'W_i': '{:.4f}',
                                    'ER': '{:.4f}', 'W_signal': '{:.4f}', 'RelScore': '{:.4f}'
                                }), use_container_width=True)
                            except Exception:
                                pass

                            # Render a small boxed, monospace block showing the numeric calculation
                            try:
                                lines = []
                                lines.append("Detailed Calculation (this scenario):")
                                if ER is not None:
                                    lines.append(f"ER = mean(W_i) = {ER:.4f}")
                                else:
                                    lines.append("ER = (could not compute)")

                                if W_signal is not None:
                                    # Show the expansion of the min(...) expression
                                    sr = sanity_ratio if sanity_ratio is not None else 'N/A'
                                    lines.append(f"W_signal = min(sanity_ratio, 3) / 3 = min({sr}, 3) / 3 = {W_signal:.4f}")
                                else:
                                    lines.append("W_signal = (sanity_ratio not available)")

                                if ER is not None and W_signal is not None and final_score is not None:
                                    lines.append(f"reliability_score = ER * W_signal = {ER:.4f} * {W_signal:.4f} = {float(final_score):.4f}")
                                elif final_score is not None:
                                    lines.append(f"reliability_score = {float(final_score):.4f} (precomputed)")
                                else:
                                    lines.append("reliability_score = (could not compute)")

                                # Bucket range (prefer batch-derived quantiles if available)
                                try:
                                    ranges = None
                                    if "batch_reliability_results" in st.session_state and dataset_name in st.session_state.batch_reliability_results:
                                        try:
                                            r_ref = st.session_state.batch_reliability_results[dataset_name]["reliability_score"]
                                            q1, q2, q3 = pd.Series(r_ref).quantile([0.25, 0.5, 0.75])
                                            ranges = {
                                                'Unreliable': (float(pd.Series(r_ref).min()), float(q1)),
                                                'Questionable': (float(q1), float(q2)),
                                                'Moderately Reliable': (float(q2), float(q3)),
                                                'Reliable': (float(q3), float(pd.Series(r_ref).max())),
                                            }
                                        except Exception:
                                            ranges = None

                                    if ranges is None:
                                        ranges = {
                                            'Reliable': (0.60, 1.0),
                                            'Moderately Reliable': (0.40, 0.60),
                                            'Questionable': (0.20, 0.40),
                                            'Unreliable': (0.0, 0.20),
                                        }

                                    if bucket in ranges:
                                        lo, hi = ranges[bucket]
                                        lines.append(f"Bucket: {bucket} (range {lo:.2f} - {hi:.2f})")
                                except Exception:
                                    pass

                                # Build HTML boxed monospace block
                                safe_html = "\n".join([str(x) for x in lines])
                                box_html = (
                                    "<div style='border:1px solid #ddd; padding:12px; background:#f7f7f9; "
                                    "border-radius:6px; font-family:monospace; font-size:14px;'>"
                                    f"<pre style='margin:0; white-space:pre-wrap;'>{safe_html}</pre>"
                                    "</div>"
                                )
                                st.markdown(box_html, unsafe_allow_html=True)
                            except Exception:
                                # Fallback to simple markdown list
                                if ER is not None:
                                    st.markdown(f"- ER = mean(W_i) = {ER:.4f}")
                                if W_signal is not None:
                                    st.markdown(f"- W_signal = min(sanity_ratio, 3) / 3 = {W_signal:.4f}")
                                if final_score is not None:
                                    st.markdown(f"- reliability_score = {float(final_score):.4f}")
                        except Exception:
                            pass
                except Exception as e_rel:
                    st.info(f"Reliability data not available: {e_rel}")
            
            # -------------------------
            # Feature-wise Reliability Construction (new user-requested section)
            # -------------------------
            if enable_reliability:
                try:
                    st.subheader("Feature-wise Reliability Construction")
                    # Get reliability data from session or recompute if needed

                    # Prefer the extended reliability_df from the row-specific calculation
                    _rank_df = None
                    if 'reliability_df' in locals() and reliability_df is not None:
                        _rank_df = reliability_df.copy()
                    elif 'rank_df' in locals() and rank_df is not None:
                        _rank_df = rank_df.copy()
                    else:
                        # Try session-state persisted results (support both dataset name and safe name)
                        try:
                            rr = st.session_state.get('reliability_results', {}) or {}
                            safe_dataset_name = str(dataset_name).replace(' ', '_').replace('.csv', '')
                            if dataset_name in rr:
                                _rank_df = rr[dataset_name].copy()
                            elif safe_dataset_name in rr:
                                _rank_df = rr[safe_dataset_name].copy()
                        except Exception:
                            _rank_df = None

                    # If still not available, attempt to load most recent persisted CSV
                    if _rank_df is None:
                        try:
                            import glob, os
                            files = glob.glob(os.path.join('results', 'reliability_table_*.csv'))
                            if files:
                                files = sorted(files, key=os.path.getmtime, reverse=True)
                                _rank_df = pd.read_csv(files[0])
                        except Exception:
                            _rank_df = None

                    if _rank_df is None or getattr(_rank_df, 'empty', True):
                        st.info("No feature rank/stability data available to build reliability table.")
                    else:
                        # Check if we have the extended columns (SHAP-based)
                        has_shap_cols = all(c in _rank_df.columns for c in ['shap_value', 'abs_shap', 'local_share', 'local_weighted_W'])
                        
                        if has_shap_cols:
                            # Display the full extended table
                            display_cols = ['feature', 'avg_rank', 'std_rank', 'W_i', 'shap_value', 'abs_shap', 'local_share', 'local_weighted_W']
                            df_table = _rank_df[[c for c in display_cols if c in _rank_df.columns]].copy()
                            df_table = df_table.rename(columns={'feature': 'Feature'})
                        else:
                            # Fallback: compute basic table without SHAP columns
                            if 'feature' not in _rank_df.columns:
                                _rank_df = _rank_df.reset_index().rename(columns={'index': 'feature'})
                            
                            _rank_df['avg_rank'] = pd.to_numeric(_rank_df['avg_rank'], errors='coerce')
                            _rank_df['std_rank'] = pd.to_numeric(_rank_df['std_rank'], errors='coerce')
                            
                            df_table = pd.DataFrame({
                                'Feature': _rank_df['feature'].astype(str),
                                'avg_rank': _rank_df['avg_rank'],
                                'std_rank': _rank_df['std_rank'],
                            })
                            df_table['T1 (1/(1+avg))'] = 1.0 / (1.0 + df_table['avg_rank'])
                            df_table['T2 (1/(1+std))'] = 1.0 / (1.0 + df_table['std_rank'])
                            df_table['W_i'] = df_table['T1 (1/(1+avg))'] * df_table['T2 (1/(1+std))']

                        # Expose avg_rank and std_rank variables for compatibility with surrounding code
                        avg_rank = df_table['avg_rank']
                        std_rank = df_table['std_rank']

                        # Display per-feature table
                        st.dataframe(df_table, use_container_width=True)

                        # -------------------------
                        # Reliability Summary (small table)
                        # -------------------------
                        try:
                            # Prefer values from reliability_metrics (which has row-specific score)
                            if 'reliability_metrics' in locals() and reliability_metrics is not None:
                                row_rel_score = reliability_metrics.get('reliability_score')
                                global_rel = reliability_metrics.get('global_reliability')
                                ER = reliability_metrics.get('er')
                                w_signal = reliability_metrics.get('w_signal')
                                _sanity = sanity_ratio if 'sanity_ratio' in locals() else None
                            else:
                                # Fallback: compute from df_table
                                ER = float(df_table['W_i'].mean()) if 'W_i' in df_table.columns and not df_table['W_i'].isna().all() else None
                                
                                # Get sanity_ratio
                                _sanity = None
                                if 'sanity_ratio' in locals() and sanity_ratio is not None:
                                    try:
                                        _sanity = float(sanity_ratio)
                                    except Exception:
                                        _sanity = None
                                else:
                                    try:
                                        rr_map = st.session_state.get('reliability_ratios', {}) or {}
                                        if dataset_name in rr_map:
                                            _sanity = float(rr_map[dataset_name])
                                        else:
                                            safe_dataset_name = str(dataset_name).replace(' ', '_').replace('.csv', '')
                                            if safe_dataset_name in rr_map:
                                                _sanity = float(rr_map[safe_dataset_name])
                                    except Exception:
                                        _sanity = None
                                
                                w_signal = min(float(_sanity) / 3.0, 1.0) if _sanity is not None and not pd.isna(_sanity) else None
                                global_rel = float(ER * w_signal) if ER is not None and w_signal is not None else None
                                
                                # Compute row-specific if we have local_weighted_W
                                if 'local_weighted_W' in df_table.columns:
                                    row_rel_score = float(w_signal * df_table['local_weighted_W'].sum()) if w_signal is not None else None
                                else:
                                    row_rel_score = global_rel

                            summary_df = pd.DataFrame({
                                'metric': [
                                    'Row Reliability Score (this instance)',
                                    'Global Reliability (ER * W_signal)',
                                    'ER (mean of W_i)',
                                    'Sanity Ratio',
                                    'W_signal (sanity/3 clipped to 1)',
                                ],
                                'value': [row_rel_score, global_rel, ER, _sanity, w_signal]
                            })

                            st.markdown('**Reliability Summary**')
                            st.dataframe(summary_df, use_container_width=True)
                        except Exception as e_sum:
                            st.warning(f"Could not compute reliability summary: {e_sum}")

                        # Show bucket ranges and highlight selected bucket based on row_rel_score
                        try:
                            # Prefer batch-derived quantile buckets if available
                            buckets = None
                            if "batch_reliability_results" in st.session_state and dataset_name in st.session_state.batch_reliability_results:
                                try:
                                    r_ref = st.session_state.batch_reliability_results[dataset_name]["reliability_score"]
                                    q1, q2, q3 = pd.Series(r_ref).quantile([0.25, 0.5, 0.75])
                                    buckets = [
                                        {"Bucket": "Unreliable", "Min": float(pd.Series(r_ref).min()), "Max": float(q1)},
                                        {"Bucket": "Questionable", "Min": float(q1), "Max": float(q2)},
                                        {"Bucket": "Moderately Reliable", "Min": float(q2), "Max": float(q3)},
                                        {"Bucket": "Reliable", "Min": float(q3), "Max": float(pd.Series(r_ref).max())},
                                    ]
                                except Exception:
                                    buckets = None

                            if buckets is None:
                                # Fallback to sensible defaults (less aggressive than previous fixed cutoffs)
                                buckets = [
                                    {"Bucket": "Reliable", "Min": 0.60, "Max": 1.0},
                                    {"Bucket": "Moderately Reliable", "Min": 0.40, "Max": 0.60},
                                    {"Bucket": "Questionable", "Min": 0.20, "Max": 0.40},
                                    {"Bucket": "Unreliable", "Min": 0.0, "Max": 0.20},
                                ]

                            bdf = pd.DataFrame(buckets)

                            sel_val = None
                            try:
                                if 'row_rel_score' in locals() and row_rel_score is not None and not pd.isna(row_rel_score):
                                    sel_val = float(row_rel_score)
                            except Exception:
                                sel_val = None

                            if sel_val is not None:
                                bdf['Selected'] = bdf.apply(lambda r: (sel_val >= r['Min'] and sel_val < r['Max']), axis=1)
                            else:
                                bdf['Selected'] = False

                            def _highlight_selected(row):
                                return ['background-color: #d4edda' if row.Selected else '' for _ in row]

                            try:
                                styled = bdf.style.format({"Min": "{:.2f}", "Max": "{:.2f}"}).apply(_highlight_selected, axis=1)
                                st.markdown("**Reliability Bucket Ranges:**")
                                st.dataframe(styled, use_container_width=True)
                            except Exception:
                                # Fallback textual list
                                st.markdown("**Reliability Bucket Ranges:**")
                                for _, r in bdf.iterrows():
                                    mark = " ✔️" if r['Selected'] else ""
                                    st.markdown(f"- {r['Bucket']} (range {r['Min']:.2f} - {r['Max']:.2f}){mark}")
                        except Exception:
                            pass

                except Exception as _err:
                    st.warning(f"Could not build feature-wise reliability tables: {_err}")
            
            # -------------------------
            # Display Waterfall Plot and AI Generated Explanation
            # -------------------------
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Waterfall Plot")
                st.caption("How each feature pushes the prediction from the base value.")
                fig, ax = plt.subplots()
                # Use max_display=10 to keep it clean
                shap.waterfall_plot(explanation, max_display=10, show=False)
                st.pyplot(fig)
                try:
                    out_fig_dir = Path(__file__).parent / "results" / "figures"
                    out_fig_dir.mkdir(parents=True, exist_ok=True)
                    safe_ds = str(dataset_name).replace(' ', '_').replace('.csv', '')
                    safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                    wf_path = out_fig_dir / f"shap_{safe_ds}_{safe_model}_waterfall_row{row_index}.png"
                    fig.savefig(wf_path, bbox_inches='tight', dpi=150)
                except Exception as e_wf:
                    try:
                        st.warning(f"Could not save waterfall PNG: {e_wf}")
                    except Exception:
                        pass
                else:
                    try:
                        # Record waterfall path in the global_shap_figures mapping
                        gmap = st.session_state.get("global_shap_figures", {})
                        dsmap = gmap.setdefault(safe_ds, {})
                        wf_map = dsmap.setdefault("waterfalls", {})
                        wf_map[f"{safe_model}_row{row_index}"] = str(wf_path)
                        dsmap["waterfalls"] = wf_map
                        gmap[safe_ds] = dsmap
                        st.session_state["global_shap_figures"] = gmap
                    except Exception:
                        pass
                plt.close(fig)

            # --- LLM Explanation in Column 2 ---
            with col2:
                st.markdown("##### AI Generated Explanation")
                st.caption("A natural language summary of the prediction.")
                
                # Always show AI commentary, but mode depends on enable_reliability flag
                # Debug toggle: when enabled, show reliability internals before LLM call (only if reliability enabled)
                debug_toggle = False
                if enable_reliability:
                    debug_toggle = st.checkbox("Show reliability debug info", key="reliability_debug_show")
                
                with st.spinner("Asking AI for an explanation..."):
                    # Try to get feature reliability metrics if available
                    feature_reliability = None
                    reliability_available = False
                    try:
                        reliability_results = st.session_state.get("reliability_results", {})
                        sanity_ratios = st.session_state.get("reliability_ratios", {})
                    
                        # Create safe name to match how reliability data is stored
                        safe_dataset_name = str(dataset_name).replace(' ', '_').replace('.csv', '')
                        
                        # Try both the original name and the safe name
                        stab_df = None
                        if dataset_name in reliability_results:
                            stab_df = reliability_results[dataset_name]
                        elif safe_dataset_name in reliability_results:
                            stab_df = reliability_results[safe_dataset_name]
                    
                        if stab_df is not None and not stab_df.empty:
                            # Build reliability dict for top features
                            feature_reliability = {}
                            for feat in explanation.feature_names[:10]:  # Top 10 features
                                if feat in stab_df['feature'].values:
                                    feat_row = stab_df[stab_df['feature'] == feat].iloc[0]
                                    feature_reliability[feat] = {
                                        'avg_rank': feat_row.get('avg_rank', 'N/A'),
                                        'std_rank': feat_row.get('std_rank', 'N/A')
                                    }
                            # Add sanity ratio if available (try both names)
                            sanity_ratio = None
                            if dataset_name in sanity_ratios:
                                sanity_ratio = sanity_ratios[dataset_name]
                            elif safe_dataset_name in sanity_ratios:
                                sanity_ratio = sanity_ratios[safe_dataset_name]
                            
                            if sanity_ratio is not None:
                                feature_reliability['_sanity_ratio'] = sanity_ratio
                            
                            if feature_reliability:
                                reliability_available = True
                                # If debug requested, show a concise snapshot
                                if debug_toggle:
                                    try:
                                        st.markdown("**Reliability debug snapshot:**")
                                        st.write({
                                            'feature_reliability_sample': feature_reliability,
                                            'sanity_ratio': sanity_ratio,
                                            'stab_df_head': stab_df.head().to_dict(orient='list') if hasattr(stab_df, 'head') else None
                                        })
                                    except Exception:
                                        pass
                    except Exception as e:
                        feature_reliability = None
                        if enable_reliability:
                            st.warning(f"Error loading reliability metrics: {e}")
                    
                    # Call LLM with appropriate mode
                    commentary, error, reliability_metrics = get_llm_explanation(
                        explanation,
                        actual_target,
                        prob_class_1,
                        feature_reliability=feature_reliability,
                        enable_row_reliability=enable_reliability
                    )
                    
                    if error:
                        st.error(f"Failed to generate explanation: {error}")
                    else:
                        st.markdown(commentary)
                        
                        # Display reliability score after explanation for reference (only if enabled and available)
                        if enable_reliability and reliability_metrics:
                            st.markdown("---")
                            st.markdown("**Reliability Metrics Reference:**")
                            col_r1, col_r2, col_r3 = st.columns(3)
                            col_r1.metric("Reliability Score", f"{reliability_metrics['reliability_score']:.3f}")
                            col_r2.metric("Classification", reliability_metrics['reliability_bucket'])
                            col_r3.metric("Sanity Ratio", f"{reliability_metrics['sanity_ratio']:.3f}")
                        elif enable_reliability and not reliability_available:
                            st.info("ℹ️ Reliability metrics not available. Run Step 5.5 (Reliability Test) to get stability analysis.")
            
            # Store local SHAP analysis results for LaTeX report generation
            try:
                # Initialize storage dict and counter if not present
                if "local_shap_analyses" not in st.session_state:
                    st.session_state.local_shap_analyses = {}
                if "analysis_counter" not in st.session_state:
                    st.session_state.analysis_counter = 0
                
                # Increment counter for new analysis
                st.session_state.analysis_counter += 1
                analysis_id = f"local_analysis_{st.session_state.analysis_counter}"
                
                # Helper function to convert numpy types to native Python types
                def convert_to_native(obj):
                    """Convert numpy types to native Python types for JSON serialization."""
                    import numpy as np
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_native(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_native(item) for item in obj]
                    else:
                        return obj

                # Build analysis record with unique ID (convert numpy types to native Python types)
                analysis_record = {
                    "analysis_id": analysis_id,
                    "dataset": dataset_name,
                    "row_index": int(row_index) if hasattr(row_index, 'item') else row_index,
                    "model_name": model_name,
                    "model_group": model_group,
                    "actual_target": int(actual_target) if hasattr(actual_target, 'item') else actual_target,
                    "predicted_prob": float(prob_class_1) if hasattr(prob_class_1, 'item') else prob_class_1,
                    "instance_features": convert_to_native(instance_features.to_dict(orient="records")[0]) if not instance_features.empty else {},
                    "ai_commentary": commentary if not error else None,
                    "waterfall_png": str(wf_path.relative_to(Path(__file__).parent)) if 'wf_path' in locals() else None,
                    "reliability_metrics": convert_to_native(reliability_metrics) if 'reliability_metrics' in locals() and reliability_metrics else None,
                }

                # Store with unique ID (persistent, no truncation)
                st.session_state.local_shap_analyses[analysis_id] = analysis_record

                # Save to disk as JSON for persistence across sessions
                try:
                    import json
                    local_analyses_dir = Path(__file__).parent / "results" / "local_analyses"
                    local_analyses_dir.mkdir(parents=True, exist_ok=True)

                    json_path = local_analyses_dir / f"{analysis_id}.json"
                    with open(json_path, 'w') as f:
                        json.dump(analysis_record, f, indent=2)
                except Exception as e:
                    # Non-fatal; continue even if disk save fails
                    st.warning(f"⚠️ Failed to save analysis to disk: {str(e)}")
                    pass

                st.success(f"✅ Analysis saved as {analysis_id}")
            except Exception:
                pass  # Non-fatal; continue without storing


            # st.markdown("---")
            # # --- START: COUNTERFACTUAL BLOCK (HIDDEN) ---
            # st.subheader("Suggested Counterfactual (minimal, feasible change)")
            #
            # if st.button("Find Counterfactual", key=f"btn_find_cf_{dataset_name}_{row_index}"):
            #     with st.spinner("Searching for a minimal counterfactual..."):
            #
            #         # 1) Prepare constraints (fast)
            #         try:
            #             import json
            #             immut = [c.strip() for c in st.session_state.get("immutable_str", "").split(",") if c.strip()]
            #         except Exception:
            #             immut = []
            #         constraints = CFConstraints(
            #             immutable=immut,
            #             lower_bounds=json.loads(st.session_state.get("lb", "{}")) if "lb" in st.session_state else {},
            #             upper_bounds=json.loads(st.session_state.get("ub", "{}")) if "ub" in st.session_state else {},
            #         )
            #
            #         # 2) Get global SHAP from Step 5 cache (fast)
            #         global_shap_cache = st.session_state.get("global_shap_dfs", {})
            #         shap_global_df = global_shap_cache.get(dataset_name)
            #
            #         if shap_global_df is None:
            #             st.error("Global SHAP data not found. Please run Step 5 (Global SHAP Analysis) first to enable this feature.")
            #             return # Stop the counterfactual search
            #
            #         abs_mean = None
            #         try:
            #             if "feature" in shap_global_df and "abs_mean" in shap_global_df:
            #                 abs_mean = shap_global_df.set_index("feature")["abs_mean"]
            #             else:
            #                  st.error("Cached Global SHAP data is invalid (missing 'abs_mean'). Please re-run Step 5.")
            #                  return # Stop
            #         except Exception as e:
            #             st.error(f"Could not read cached SHAP data: {e}. Please re-run Step 5.")
            #             return # Stop
            #
            #         # 3) Get direction from the current instance's explanation (fast)
            #         directions = {}
            #         try:
            #             # explanation.values is 1 x p
            #             val = explanation.values if hasattr(explanation, "values") else None
            #             if val is not None:
            #                 s = np.sign(val.reshape(-1))
            #                 directions = {f: (1 if s[i] > 0 else (-1 if s[i] < 0 else 0)) for i, f in enumerate(instance_features.columns)}
            #         except Exception as e:
            #             st.warning(f"Could not derive directions: {e}")
            #
            #         # 4) Run the search (now much faster)
            #         if abs_mean is not None and directions:
            #             cf = find_counterfactual(
            #                 model=model,
            #                 x0=instance_features,
            #                 train_sample=X_train,
            #                 shap_abs_mean=abs_mean,
            #                 directions=directions,
            #                 constraints=constraints,
            #                 target_class=1,          # flip toward class 1 by default
            #                 beam_width=20,
            #                 max_steps=30,
            #                 alpha=1.0,
            #                 beta=0.02
            #             )
            #             if cf["x_cf"] is not None:
            #                 st.success(f"Found a counterfactual with {cf['changes']} changes (objective={cf['objective']:.3f}).")
            #                 st.dataframe(pd.concat([instance_features.reset_index(drop=True), cf["x_cf"]], axis=0).assign(_row=["original","counterfactual"]).set_index("_row"))
            #                 st.caption("Interpretation: move variables in the observed directions to flip the decision with minimal change.")
            #                 st.download_button("Download counterfactual CSV", data=cf["x_cf"].to_csv(index=False), file_name=f"cf_{dataset_name}_row{row_index}.csv")
            #             else:
            #                 st.warning("No counterfactual found under current constraints and step limits. Relax bounds or increase max_steps.")
            #         else:
            #             st.info("Counterfactual search skipped (no SHAP abs-mean or directions).")
            # # --- END: COUNTERFACTUAL BLOCK (HIDDEN) ---

            # st.markdown("---")
            # # --- START: NEW INTERACTIONS & PDP BLOCK (HIDDEN) ---
            # st.subheader("Feature Interactions & Local Response")
            # 
            # # 1. Put the feature selector *outside* the button
            # feat = st.selectbox("Feature for PDP/ICE",
            #                     options=list(instance_features.columns),
            #                     key=f"pdp_select_{dataset_name}_{row_index}") # Unique key
            #
            # # 2. Put the analysis inside a button
            # if st.button("Analyze Interactions & PDP/ICE", key=f"btn_interact_{dataset_name}_{row_index}"):
            #     with st.spinner("Analyzing interactions and PDP..."):
            #         colA, colB = st.columns(2)
            #         with colA:
            #             st.markdown("**Top SHAP interactions (tree models)**")
            #             try:
            #                 pairs = shap_top_interactions_for_tree(model, X_train.sample(min(512, len(X_train))))
            #                 if pairs:
            #                     df_pairs = pd.DataFrame(pairs, columns=["feature_i","feature_j","mean_abs_interaction"])
            #                     st.dataframe(df_pairs)
            #                 else:
            #                     st.caption("Not available for this model type or failed to compute.")
            #             except Exception as e:
            #                 st.caption(f"Interaction computation failed: {e}")
            #
            #         with colB:
            #             st.markdown("**ICE/PDP Plot**")
            #             # Read the feature from the selectbox's state
            #             selected_feat = st.session_state[f"pdp_select_{dataset_name}_{row_index}"] 
            #             
            #             import matplotlib.pyplot as plt
            #             fig, ax = plt.subplots()
            #             plot_ice_pdp(ax, model, X_train, selected_feat, n_ice=100)
            #             st.pyplot(fig)
            #             try:
            #                 out_fig_dir = Path(__file__).parent / "results" / "figures"
            #                 out_fig_dir.mkdir(parents=True, exist_ok=True)
            #                 safe_ds = str(dataset_name).replace(' ', '_').replace('.csv', '')
            #                 safe_model = str(model_name).replace(' ', '_').replace('/', '_')
            #                 safe_feat = str(selected_feat).replace(' ', '_').replace('/', '_')
            #                 pdp_path = out_fig_dir / f"shap_{safe_ds}_{safe_model}_pdp_row{row_index}_{safe_feat}.png"
            #                 fig.savefig(pdp_path, bbox_inches='tight', dpi=150)
            #             except Exception as e_pdp:
            #                 try:
            #                     st.warning(f"Could not save PDP/ICE PNG: {e_pdp}")
            #                 except Exception:
            #                     pass
            #             else:
            #                 try:
            #                     # Record PDP path in the global_shap_figures mapping
            #                     gmap = st.session_state.get("global_shap_figures", {})
            #                     dsmap = gmap.setdefault(safe_ds, {})
            #                     pdp_map = dsmap.setdefault("pdp", {})
            #                     pdp_map[f"{safe_model}_row{row_index}_{safe_feat}"] = str(pdp_path)
            #                     dsmap["pdp"] = pdp_map
            #                     gmap[safe_ds] = dsmap
            #                     st.session_state["global_shap_figures"] = gmap
            #                 except Exception:
            #                     pass
            #             plt.close(fig)
            # # --- END: NEW INTERACTIONS & PDP BLOCK (HIDDEN) ---

        except Exception as e:
            st.error(f"Failed to generate local SHAP plot: {e}")
            st.exception(e) # Show full traceback
            
    def _save_model_comparison_png(self, out_dir: Path, metric: str = "AUC") -> Optional[str]:
        """
        Create and save a simple Matplotlib model-comparison PNG for the given metric
        using the `benchmark_results_df` in session state. Charts Model Groups on x-axis.
        Returns the PNG path or None.
        """
        df = st.session_state.get("benchmark_results_df")
        if df is None or df.empty:
            return None

        try:
            # pivot: index = Model Group, columns = Dataset, values = metric
            pivot = df.pivot_table(index="Model Group", columns="Dataset", values=metric)
        except Exception:
            try:
                pivot = df.groupby(["Model Group", "Dataset"]).agg({metric: "mean"}).unstack(fill_value=np.nan)
                pivot.columns = pivot.columns.get_level_values(1)
            except Exception:
                return None

        # Use an rc context so font increases affect only this export (no global side-effects)
        with plt.rc_context({
            # Increased by ~30% from previous values
            "font.size": 21,
            "axes.titlesize": 23,
            "axes.labelsize": 21,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
        }):
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in pivot.columns:
                # Remove trailing .csv from dataset labels for cleaner legends
                lbl = str(col)
                if lbl.lower().endswith('.csv'):
                    lbl = lbl[:-4]
                ax.plot(pivot.index, pivot[col], marker="o", label=lbl)

            ax.set_title(f"Model comparison ({metric})")
            ax.set_xlabel("Model Group")
            ax.set_ylabel(metric)
            ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_xticks(range(len(pivot.index)))
            # Fix for null labels: explicitly convert to string and handle None/NaN
            x_labels = []
            for x in pivot.index:
                if pd.isna(x) or x is None:
                    x_labels.append("Unknown")
                else:
                    x_labels.append(str(x))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            fig.tight_layout()

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"model_comparison_{metric}.png"
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return str(out_path)
        except Exception:
            plt.close(fig)
            return None

    def _summarise_models_from_code(self, df: pd.DataFrame, client) -> str:
        """
        Inspect `models.MODELS` and the benchmark DataFrame to build a small
        code-context text block, then ask the LLM to summarise model families.
        Returns the plain-text LLM response or an empty string on failure.
        """
        try:
            import models as models_module
            from models import MODELS as MODELS_DICT
        except Exception as e:
            st.warning(f"Could not import models.py for model summarization: {e}")
            return ""

        # Collect distinct model groups from the benchmark DF
        groups = []
        try:
            if "Model Group" in df.columns:
                groups = [str(x) for x in pd.unique(df["Model Group"].dropna())]
        except Exception:
            groups = []

        # Collect top benchmark models by AUC (up to ~8 unique names)
        top_models = []
        try:
            if "Benchmark Model" in df.columns and "AUC" in df.columns:
                tmp = df[["Benchmark Model", "AUC"]].dropna()
                tmp = tmp.sort_values("AUC", ascending=False)
                for name in tmp["Benchmark Model"].astype(str).tolist():
                    if name not in top_models:
                        top_models.append(name)
                    if len(top_models) >= 8:
                        break
            elif "Benchmark Model" in df.columns:
                top_models = list(pd.unique(df["Benchmark Model"].astype(str)))[:8]
        except Exception:
            top_models = []

        # Find builder functions in MODELS_DICT that match the top model names
        snippets = []
        try:
            for g in (groups or list(MODELS_DICT.keys())):
                group_dict = MODELS_DICT.get(g, {})
                for model_name, builder in group_dict.items():
                    if model_name in top_models:
                        try:
                            src = inspect.getsource(builder)
                        except Exception:
                            # Fallback: try to get __call__ source or skip
                            try:
                                src = inspect.getsource(builder.__call__)
                            except Exception:
                                src = f"# source not available for {g}/{model_name}\n"
                        snippets.append(f"### {g}/{model_name}\n{src}")

            # If no snippets found using groups, try a global search across MODELS_DICT
            if not snippets:
                for g, group_dict in MODELS_DICT.items():
                    for model_name, builder in group_dict.items():
                        if model_name in top_models:
                            try:
                                src = inspect.getsource(builder)
                            except Exception:
                                src = f"# source not available for {g}/{model_name}\n"
                            snippets.append(f"### {g}/{model_name}\n{src}")
        except Exception as e:
            st.warning(f"Error while extracting builder source: {e}")

        snippet_text = "\n\n".join(snippets)

        # Prepare LLM call
        try:
            system_msg = (
                "You are an assistant that writes concise, accurate descriptions of machine-learning model families for academic papers."
            )
            user_msg = (
                "The following code snippets are builder functions that construct models used in a credit-risk benchmarking study. "
                "Group these builders into algorithm families (e.g., logistic regression, regularised logistic regression, random forests, "
                "bagged CART, AdaBoost, gradient boosting, KNN, neural nets, XGBoost, LightGBM, etc.). "
                "For each family, produce a single bullet that names the family, lists the internal labels (Model Group / Benchmark Model), "
                "and provides a 1–3 sentence academic-style description of the family's behavior and suitability for credit-risk modelling.\n\n"
            )
            if snippet_text:
                user_msg = user_msg + "CODE SNIPPETS:\n\n" + snippet_text
            else:
                user_msg = user_msg + "(No code snippets were found or accessible.)"

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=600,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"Model-family summarization failed: {e}")
            return ""
    
    
    def _render_step_7_5_batch_reliability(self):
        """
        Renders Step 7.5: Batch Reliability Computation for all rows in selected datasets.
        """
        from pathlib import Path
        from io import StringIO
        
        st.header("📊 Step 7.5: Batch Reliability Computation")
        st.markdown("Compute predictions and row-specific reliability scores for all rows in your datasets.")
        
        # Check prerequisites
        if st.session_state.get("benchmark_results_df") is None:
            st.info("Run Step 4 to identify benchmark models before computing batch reliability.")
            return
        
        all_results = st.session_state.get("results", {})
        if not all_results:
            st.info("Run Step 3 to train models before computing batch reliability.")
            return
        
        # Dataset selection
        dataset_name = st.selectbox(
            "Select a dataset for batch reliability computation:",
            st.session_state.selected_datasets,
            key="batch_reliability_dataset_select"
        )
        
        if not dataset_name:
            st.info("Please select a dataset.")
            return
        
        # Load the dataset
        try:
            if dataset_name not in st.session_state.full_dfs:
                with st.spinner(f"Loading {dataset_name}..."):
                    fileobj = st.session_state.uploaded_files_map[dataset_name]
                    fileobj.seek(0)
                    from io import StringIO
                    st.session_state.full_dfs[dataset_name] = pd.read_csv(StringIO(fileobj.getvalue().decode("utf-8")))
            
            df = st.session_state.full_dfs[dataset_name]
            target_col = st.session_state.target_column
            
            # Get benchmark model info
            benchmark_df = st.session_state.benchmark_results_df
            benchmark_row = benchmark_df[benchmark_df['Dataset'] == dataset_name]
            
            if benchmark_row.empty:
                st.error(f"No benchmark model found for {dataset_name}. Please run Step 4.")
                return
            
            model_group = benchmark_row.iloc[0]['Model Group']
            model_name = benchmark_row.iloc[0]['Benchmark Model']
            
            st.info(f"Using benchmark model: **{model_name}** (from group: {model_group})")
            
            # Get model and training data
            model_data = all_results.get(dataset_name, {})
            model = model_data.get('models', {}).get(model_group, {}).get(model_name)
            X_train = model_data.get('data', {}).get('X_train')
            
            if model is None or X_train is None:
                st.error("Fitted model or training data not found. Please re-run Step 3.")
                return
            
            # Get reliability data
            try:
                n_trials = int(st.session_state.get("rel_n_trials", 1))
                bg_size = int(st.session_state.get("rel_n_bg", 20))
                model_block = all_results.get(dataset_name, {}).get('data', {})
                X_test = model_block.get('X_test')
                y_train = model_block.get('y_train')
                
                # Reuse the compute_or_load function from Step 7
                rank_df, sanity_ratio, summary_text = self._compute_or_load_reliability_for_dataset(
                    dataset_name, model, X_train, X_test, y_train, n_trials=n_trials, bg_size=bg_size
                )
                
                if rank_df is None or rank_df.empty:
                    st.error("Reliability data not available. Please run Step 5.5 (Reliability Tests) first.")
                    return
                if sanity_ratio is None:
                    st.error("Sanity ratio not available. Please run Step 5.5 (Reliability Tests) first.")
                    return
            except Exception as e:
                st.error(f"Could not load reliability data: {e}")
                return
            
            # Threshold input
            col_thresh1, col_thresh2 = st.columns([1, 2])
            with col_thresh1:
                threshold = st.number_input(
                    "Classification threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Probability threshold for predicting default (class 1)",
                    key=f"batch_threshold_{dataset_name}"
                )
            
            # Computation button
            run_batch = st.button("Run reliability for all rows", key=f"batch_reliability_btn_{dataset_name}")
            
            if run_batch:
                try:
                    with st.spinner("Computing batch predictions and reliability scores..."):
                        # Get the full dataset (without target column)
                        X_all = df.drop(columns=[target_col])
                        
                        # Get true labels if available
                        y_all = None
                        if target_col in df.columns:
                            y_all = df[target_col].values
                        
                        # Compute reliability for each row
                        st.info(f"Computing SHAP and reliability for {len(X_all)} rows... This may take a few minutes.")
                        
                        results_list = []
                        progress_bar = st.progress(0)
                        
                        for idx in range(len(X_all)):
                            # Get single row as DataFrame
                            row_df = X_all.iloc[[idx]]
                            
                            # Call the unified reliability function
                            result = compute_single_row_reliability(
                                model=model,
                                X_train=X_train,
                                row_df=row_df,
                                reliability_df=rank_df,
                                sanity_ratio=sanity_ratio,
                                threshold=threshold,
                                max_bg=200
                            )
                            results_list.append(result)
                            
                            # Update progress every 10 rows
                            if (idx + 1) % 10 == 0 or idx == len(X_all) - 1:
                                progress_bar.progress((idx + 1) / len(X_all))
                        
                        progress_bar.empty()
                        
                        # Build results DataFrame
                        results_df = pd.DataFrame(results_list)
                        
                        # Add actual labels if available
                        if y_all is not None:
                            results_df["actual_default"] = y_all
                        
                        # Create quantile-based buckets
                        r = results_df["reliability_score"]
                        q1, q2, q3 = r.quantile([0.25, 0.5, 0.75])
                        
                        def assign_bucket_unified(x):
                            if x >= q3:
                                return "Reliable"
                            elif x >= q2:
                                return "Moderately Reliable"
                            elif x >= q1:
                                return "Questionable"
                            else:
                                return "Unreliable"

                        results_df["reliability_bucket"] = r.apply(assign_bucket_unified)

                        # Create bucket definition table
                        buckets_df = pd.DataFrame({
                            "bucket": ["Unreliable", "Questionable", "Moderately Reliable", "Reliable"],
                            "lower_bound": [r.min(), q1, q2, q3],
                            "upper_bound": [q1, q2, q3, r.max()],
                            "description": [
                                "Bottom 25% of reliability scores (least reliable)",
                                "25–50% quantile (questionable)",
                                "50–75% quantile (moderately reliable)",
                                "Top 25% of reliability scores (most reliable)",
                            ],
                        })
                        
                        # Store in session state
                        if "batch_reliability_results" not in st.session_state:
                            st.session_state.batch_reliability_results = {}
                        st.session_state.batch_reliability_results[dataset_name] = results_df
                        
                        if "batch_reliability_buckets" not in st.session_state:
                            st.session_state.batch_reliability_buckets = {}
                        st.session_state.batch_reliability_buckets[dataset_name] = buckets_df
                        
                        # Save to file using ResultManager
                        safe_name = dataset_name.replace('.csv', '').replace(' ', '_')
                        from pathlib import Path

                        try:
                            if self.result_mgr:
                                saved_paths = self.result_mgr.save_batch_reliability(
                                    dataset_name=dataset_name,
                                    results_df=results_df,
                                    buckets_df=buckets_df
                                )
                                output_path = saved_paths.get('excel')
                            else:
                                output_filename = f"batch_reliability_{safe_name}.xlsx"
                                output_path = Path("results") / "batch" / output_filename
                                output_path.parent.mkdir(parents=True, exist_ok=True)

                                with pd.ExcelWriter(str(output_path), engine="xlsxwriter") as writer:
                                    results_df.to_excel(writer, sheet_name="row_scores", index=False)
                                    buckets_df.to_excel(writer, sheet_name="buckets", index=False)

                            # Store path for download
                            if "batch_reliability_excel_paths" not in st.session_state:
                                st.session_state.batch_reliability_excel_paths = {}
                            st.session_state.batch_reliability_excel_paths[dataset_name] = output_path

                        except Exception as e_excel:
                            st.warning(f"Could not write Excel file: {e_excel}.")
                        
                        st.success(f"✅ Computed reliability scores for {len(results_df)} rows!")
                
                except Exception as e_batch:
                    st.error(f"Batch computation failed: {e_batch}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Display results if available
            results_df = None
            buckets_df = None
            if "batch_reliability_results" in st.session_state:
                results_df = st.session_state.batch_reliability_results.get(dataset_name)
            if "batch_reliability_buckets" in st.session_state:
                buckets_df = st.session_state.batch_reliability_buckets.get(dataset_name)
            
            if results_df is not None and not results_df.empty:
                st.markdown("---")
                st.markdown("### Batch Reliability Results")
                
                # Display results table
                st.markdown("**Batch Predictions and Reliability Scores:**")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.markdown("**Summary Statistics:**")
                summary = results_df["reliability_score"].describe()
                st.dataframe(summary.to_frame().T, use_container_width=True)
                
                # Bucket distribution
                st.markdown("**Distribution by Reliability Bucket:**")
                bucket_counts = results_df["reliability_bucket"].value_counts()
                st.bar_chart(bucket_counts)
                
                # Bucket definitions
                if buckets_df is not None:
                    st.markdown("**Bucket Definitions (dataset-specific quantiles):**")
                    st.dataframe(buckets_df, use_container_width=True)
                
                # Download buttons
                safe_name = dataset_name.replace('.csv', '').replace(' ', '_')
                
                # Excel download
                excel_path = st.session_state.get("batch_reliability_excel_paths", {}).get(dataset_name)
                if excel_path and Path(excel_path).exists():
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Excel (scores + buckets)",
                            data=f,
                            file_name=f"batch_reliability_{safe_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                
                # CSV download
                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download batch reliability CSV",
                    data=csv,
                    file_name=f"batch_reliability_{safe_name}.csv",
                    mime="text/csv",
                )
                
                # Diagnostics expander
                diag_key = f"reliability_diag_{dataset_name}"
                with st.expander("Reliability Statistical Diagnostics", expanded=False):
                    st.markdown(
                        "Run statistical tests to evaluate how well `reliability_score` "
                        "separates correct vs incorrect predictions."
                    )
                    
                    if st.button("Run diagnostics", key=f"run_diag_step75_{dataset_name}"):
                        try:
                            st.session_state[diag_key] = run_reliability_diagnostics(results_df)
                        except Exception as e_diag:
                            st.error(f"Diagnostics failed: {e_diag}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    if diag_key in st.session_state:
                        diag = st.session_state[diag_key]
                        
                        cols = st.columns(3)
                        cols[0].metric("Correct predictions", f"{diag.get('n_correct', 'N/A')}")
                        cols[1].metric("Wrong predictions", f"{diag.get('n_wrong', 'N/A')}")
                        auc = diag.get("roc_auc")
                        cols[2].metric("ROC AUC (reliability)", f"{auc:.3f}" if auc is not None else "N/A")
                        
                        st.markdown("**Statistical test results**")
                        st.write({
                            "KS statistic": diag.get("ks_stat"),
                            "KS p-value": diag.get("ks_pvalue"),
                            "Mann-Whitney U": diag.get("mw_stat"),
                            "Mann-Whitney p-value": diag.get("mw_pvalue"),
                            "Logistic coef": diag.get("logit_coef"),
                            "Logistic p-value": diag.get("logit_pvalue"),
                            "Logistic p-value (Wald)": diag.get("logit_pvalue_wald"),
                            "Pseudo R2 (McFadden)": diag.get("logit_pseudo_r2"),
                        })
                        
                        # Interpretation guidance
                        interp_lines = []
                        if auc is not None:
                            if auc >= 0.8:
                                interp_lines.append("ROC AUC indicates strong separation (>= 0.8).")
                            elif auc >= 0.7:
                                interp_lines.append("ROC AUC indicates moderate separation (0.7–0.8).")
                            elif auc >= 0.6:
                                interp_lines.append("ROC AUC indicates weak separation (0.6–0.7).")
                            else:
                                interp_lines.append("ROC AUC indicates little to no separation (< 0.6).")
                        
                        pval = diag.get("logit_pvalue")
                        if pval is not None:
                            if pval < 0.01:
                                interp_lines.append("Logistic regression: coefficient highly significant (p < 0.01).")
                            elif pval < 0.05:
                                interp_lines.append("Logistic regression: coefficient significant (p < 0.05).")
                            else:
                                interp_lines.append("Logistic regression: coefficient not statistically significant (p >= 0.05).")
                        
                        if interp_lines:
                            st.markdown("**Interpretation:**")
                            for L in interp_lines:
                                st.write(f"- {L}")
                        
                        if st.button("Clear diagnostics", key=f"clear_diag_step75_{dataset_name}"):
                            del st.session_state[diag_key]
        
        except Exception as e:
            st.error(f"Failed to process dataset {dataset_name}: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    def _render_step_7_report_generation(self):
        """
        Renders the UI for Step 7: Two-Phase Results Section Generation.
        Phase 1: Assemble artifacts (instant, no LLM)
        Phase 2: Generate narratives (optional, uses LLM)
        """
        st.header("📝 Step 8: Generate LaTeX Report")
        
        st.markdown("""
        Generate an academic-quality **Results section** for your research paper in two phases:
        
        **Phase 1 - Assemble Artifacts** (instant, no cost):
        - Organizes all analysis outputs into a structured Results section
        - Creates tables and figures in academic LaTeX format
        - No LLM calls - immediate preview available
        
        **Phase 2 - Add Narratives** (optional, uses LLM):
        - Generates concise academic commentary for each subsection
        - Interprets results and provides insights
        - Toggle on/off to control cost and customization
        """)
        
        # Import managers
        try:
            from report_manager import get_report_manager
            from result_manager import get_result_manager
        except ImportError:
            st.error("ReportManager or ResultManager module not found. Ensure report_manager.py and result_manager.py are in the project directory.")
            return
        
        # Initialize managers
        report_mgr = get_report_manager("reports")
        result_mgr = get_result_manager("results")
        
        # Check available content
        with st.spinner("Scanning results folder..."):
            manifest = result_mgr.prepare_report_manifest()
        
        # Extract datasets from manifest
        datasets = []
        if manifest.get('eda'):
            datasets.extend(list(manifest['eda'].keys()))
        if manifest.get('reliability'):
            for ds in manifest['reliability'].keys():
                if ds not in datasets:
                    datasets.append(ds)
        if manifest.get('batch_reliability'):
            for ds in manifest['batch_reliability'].keys():
                if ds not in datasets:
                    datasets.append(ds)
        
        # Display what's available
        st.subheader("📊 Available Artifacts")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Datasets", len(datasets))
            if datasets:
                st.write("**Datasets:**")
                for ds in datasets:
                    st.write(f"  • {ds}")
        
        with col2:
            artifact_count = 0
            # Count EDA artifacts
            if manifest.get('eda'):
                artifact_count += len(manifest['eda'])
            # Count benchmark artifacts
            if manifest.get('benchmarks', {}).get('csv'):
                artifact_count += 1
            # Count SHAP artifacts
            if manifest.get('shap_plots'):
                artifact_count += len(manifest['shap_plots'])
            # Count reliability artifacts
            if manifest.get('reliability'):
                artifact_count += len(manifest['reliability'])
            if manifest.get('batch_reliability'):
                artifact_count += len(manifest['batch_reliability'])
            # Count local analyses
            if manifest.get('local_analyses'):
                artifact_count += len(manifest['local_analyses'])
            st.metric("Total Artifacts", artifact_count)
        
        with col3:
            st.write("**Available Analyses:**")
            if manifest.get('eda'):
                st.write("✅ EDA")
            if manifest.get('benchmarks', {}).get('csv'):
                st.write("✅ Benchmarks")
            if manifest.get('shap_plots'):
                st.write("✅ SHAP")
            if manifest.get('reliability') or manifest.get('batch_reliability'):
                st.write("✅ Reliability")
            if manifest.get('local_analyses'):
                st.write("✅ Local Examples")
        
        st.markdown("---")
        
        # ==================== PHASE 1: ASSEMBLE ARTIFACTS ====================
        st.subheader("🔧 Phase 1: Assemble Results Artifacts")
        
        st.markdown("""
        Click below to organize all analysis outputs into a structured Results section.
        This creates tables and figures in academic LaTeX format **instantly** (no LLM calls).
        """)
        
        if st.button("📦 Assemble Artifacts", type="primary", use_container_width=True, key="assemble_artifacts_btn"):
            # Check if we have ANY artifacts (not just datasets)
            has_artifacts = (
                manifest.get('eda') or
                manifest.get('benchmarks', {}).get('csv') or
                manifest.get('shap_plots') or
                manifest.get('reliability') or
                manifest.get('batch_reliability') or
                manifest.get('local_analyses')
            )

            if not has_artifacts:
                st.warning("No analysis results found. Please complete Steps 1-6 first.")
                return
            
            with st.spinner("Organizing artifacts into Results section..."):
                try:
                    assembled = report_mgr.assemble_results_artifacts(
                        result_manager=result_mgr,
                        include_placeholders=True  # Include for Phase 2
                    )
                    
                    st.session_state.assembled_results = assembled
                    st.success("✅ Artifacts assembled successfully!")
                    
                    # Show summary
                    summary = assembled.get('artifact_summary', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tables Created", summary.get('total_tables', 0))
                    with col2:
                        st.metric("Figures Included", summary.get('total_figures', 0))
                    with col3:
                        st.metric("Placeholders", len(assembled.get('placeholders', [])))
                    
                except Exception as e:
                    st.error(f"Error assembling artifacts: {str(e)}")
                    return
        
        # Display assembled results
        if 'assembled_results' in st.session_state:
            st.markdown("---")
            st.subheader("📄 Assembled Results Section")

            assembled = st.session_state.assembled_results

            # PDF Compilation Section (outside tabs to avoid navigation issues)
            import shutil
            if shutil.which('pdflatex') is not None:
                st.markdown("### 🔨 PDF Compilation")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("Compile the LaTeX source to PDF. The compiled PDF will appear in the 'PDF Preview' tab below.")
                with col2:
                    compile_clicked = st.button("🔨 Compile LaTeX to PDF", use_container_width=True, key="compile_pdf_btn_top")

                if compile_clicked:
                    progress_placeholder = st.empty()
                    progress_placeholder.info("⏳ Preparing LaTeX document...")

                    try:
                        import subprocess
                        import tempfile
                        from pathlib import Path

                        # Create a complete LaTeX document with necessary packages
                        latex_full = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\usepackage{placeins}
\usepackage{adjustbox}

\title{Credit Risk Modeling Results}
\author{Research Team}
\date{\today}

\begin{document}
\maketitle

""" + assembled.get('latex', '') + r"""

\end{document}
"""

                        # Create temporary directory for compilation
                        with tempfile.TemporaryDirectory() as tmpdir:
                            tmppath = Path(tmpdir)
                            tex_file = tmppath / "results.tex"

                            # Write LaTeX file
                            progress_placeholder.info("⏳ Writing LaTeX file...")
                            with open(tex_file, 'w', encoding='utf-8') as f:
                                f.write(latex_full)

                            # Copy all figures to temp directory
                            progress_placeholder.info("⏳ Copying figures...")
                            figures_dir = Path("results/figures")
                            if figures_dir.exists():
                                for fig in figures_dir.glob("*.png"):
                                    import shutil as sh
                                    sh.copy(fig, tmppath / fig.name)

                            # Run pdflatex (twice for references)
                            progress_placeholder.info("⏳ Compiling PDF (pass 1/2)... This may take 30-60 seconds on first run.")

                            for i in range(2):
                                result = subprocess.run(
                                    ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'results.tex'],
                                    cwd=tmppath,
                                    capture_output=True,
                                    text=True,
                                    timeout=120  # Increased timeout for package installation
                                )
                                if i == 0:
                                    progress_placeholder.info("⏳ Compiling PDF (pass 2/2)...")

                            pdf_file = tmppath / "results.pdf"

                            if pdf_file.exists():
                                # Read PDF and store in session state
                                progress_placeholder.info("⏳ Loading PDF...")
                                with open(pdf_file, 'rb') as f:
                                    pdf_data = f.read()

                                st.session_state.compiled_pdf = pdf_data
                                progress_placeholder.success("✅ PDF compiled successfully! Check the 'PDF Preview' tab below.")

                                # Also save log for debugging
                                log_file = tmppath / "results.log"
                                if log_file.exists():
                                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                        st.session_state.latex_log = f.read()
                                
                                # Rerun to refresh UI and show PDF
                                st.rerun()
                            else:
                                progress_placeholder.empty()
                                st.error("❌ PDF compilation failed. Check LaTeX log below.")
                                log_file = tmppath / "results.log"
                                if log_file.exists():
                                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                        log_content = f.read()
                                    with st.expander("View LaTeX Log (Click to Debug)"):
                                        st.text(log_content[-5000:])  # Last 5000 chars

                    except subprocess.TimeoutExpired:
                        progress_placeholder.empty()
                        st.error("❌ PDF compilation timed out (>120s). This may happen if MiKTeX is installing packages. Try again or compile manually.")
                    except Exception as e:
                        progress_placeholder.empty()
                        st.error(f"❌ Error during compilation: {str(e)}")

                st.markdown("---")

            # Tabs for preview
            preview_tabs = st.tabs(["📝 Markdown Preview", "📜 LaTeX Source", "📄 PDF Preview", "📊 Artifact Summary"])
            
            with preview_tabs[0]:
                st.markdown("**Markdown Preview:**")
                with st.expander("View Full Markdown", expanded=True):
                    st.markdown(assembled.get('markdown', ''))
            
            with preview_tabs[1]:
                st.markdown("**LaTeX Source Code:**")
                with st.expander("View LaTeX Code", expanded=False):
                    st.code(assembled.get('latex', ''), language='latex')
            
            with preview_tabs[2]:
                st.markdown("**PDF Preview:**")

                # Display PDF if available (or show instructions if not compiled yet)
                if 'compiled_pdf' in st.session_state:
                    st.markdown("---")
                    st.success("📄 PDF ready for download!")

                    # Download button
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=st.session_state.compiled_pdf,
                        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )

                    # Optional: Show PDF in expander (instead of iframe which can hang)
                    with st.expander("🔍 Preview PDF (may not work in all browsers)", expanded=True):
                        try:
                            import base64
                            base64_pdf = base64.b64encode(st.session_state.compiled_pdf).decode('utf-8')
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)
                            st.caption("💡 If preview doesn't render properly, use the download button above.")
                        except Exception as e:
                            st.warning(f"PDF preview not available: {e}. Please use download button.")

                    # Show log option
                    if 'latex_log' in st.session_state:
                        with st.expander("📋 View Compilation Log"):
                            st.text(st.session_state.latex_log[-5000:])  # Last 5000 chars
                else:
                    # Show instructions when no PDF has been compiled
                    import shutil
                    if shutil.which('pdflatex') is None:
                        st.warning("⚠️ pdflatex not found on your system. Please install a LaTeX distribution (e.g., MiKTeX, TeX Live) to compile PDFs.")
                        st.info("You can still download the .tex file from the 'LaTeX Source' tab and compile it manually.")
                    else:
                        st.info("📝 No PDF compiled yet. Click the '🔨 Compile LaTeX to PDF' button above to generate the PDF.")
                        st.info("💡 Tip: The compilation may take 30-60 seconds on the first run while LaTeX packages are installed.")
            
            with preview_tabs[3]:
                st.markdown("**Artifact Breakdown:**")
                summary = assembled.get('artifact_summary', {})
                
                # EDA
                if summary.get('eda') and (summary['eda']['tables'] > 0 or summary['eda']['figures'] > 0):
                    st.write(f"**EDA:** {summary['eda']['tables']} tables, {summary['eda']['figures']} figures")
                
                # Benchmark
                if summary.get('benchmark') and (summary['benchmark']['tables'] > 0 or summary['benchmark']['figures'] > 0):
                    st.write(f"**Benchmark:** {summary['benchmark']['tables']} tables, {summary['benchmark']['figures']} figures")
                
                # SHAP
                if summary.get('shap') and (summary['shap']['tables'] > 0 or summary['shap']['figures'] > 0):
                    st.write(f"**SHAP:** {summary['shap']['tables']} tables, {summary['shap']['figures']} figures")
                
                # Reliability
                if summary.get('reliability') and (summary['reliability']['tables'] > 0 or summary['reliability']['figures'] > 0):
                    st.write(f"**Reliability:** {summary['reliability']['tables']} tables, {summary['reliability']['figures']} figures")
                
                # Local
                if summary.get('local') and (summary['local']['tables'] > 0 or summary['local']['figures'] > 0):
                    st.write(f"**Local SHAP Examples:** {summary['local']['figures']} waterfall plots")
                
                # Total summary
                total_tables = sum(s.get('tables', 0) for s in summary.values() if isinstance(s, dict))
                total_figures = sum(s.get('figures', 0) for s in summary.values() if isinstance(s, dict))
                st.markdown("---")
                st.write(f"**Grand Total:** {total_tables} tables, {total_figures} figures")
            
            # Download buttons for Phase 1 (artifact-only)
            st.markdown("**Download Options:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="⬇️ Download Markdown (Artifacts Only)",
                    data=assembled.get('markdown', ''),
                    file_name=f"results_artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    label="⬇️ Download LaTeX (Artifacts Only)",
                    data=assembled.get('latex', ''),
                    file_name=f"results_artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                    mime="text/x-tex",
                    use_container_width=True
                )
            
            # ==================== PHASE 2: GENERATE NARRATIVES ====================
            st.markdown("---")
            st.subheader("🤖 Phase 2: Add AI-Generated Narratives (Optional)")
            
            # Toggle for Phase 2
            enable_narratives = st.checkbox(
                "Enable AI Commentary",
                value=False,
                help="Generate concise academic narratives to interpret the results (uses LLM API)"
            )
            
            if enable_narratives:
                st.markdown("""
                Generate concise academic commentary for each Results subsection.
                Each narrative will be **100-180 words** of interpretive text.
                """)
                
                # LLM Configuration
                with st.expander("⚙️ LLM Configuration", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        llm_provider = st.selectbox(
                            "Provider",
                            options=["openai", "anthropic"],
                            index=0,
                            help="Select the LLM provider"
                        )
                        
                        if llm_provider == "openai":
                            default_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
                        else:
                            default_models = ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
                        
                        llm_model = st.selectbox(
                            "Model",
                            options=default_models,
                            index=0
                        )
                    
                    with col2:
                        api_key = st.text_input(
                            "API Key",
                            type="password",
                            value="",
                            help=f"Enter your {llm_provider.upper()} API key",
                            key="narrative_api_key"
                        )
                        
                        temperature = st.slider(
                            "Temperature",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.3,
                            step=0.1,
                            help="Lower = more focused, Higher = more creative"
                        )
                
                # Generate Narratives Button
                if st.button("🎨 Generate Narratives", type="secondary", use_container_width=True, key="generate_narratives_btn"):
                    if not api_key:
                        st.error("Please provide an API key to generate narratives.")
                        return
                    
                    llm_config = {
                        'provider': llm_provider,
                        'api_key': api_key,
                        'model': llm_model,
                        'temperature': temperature,
                        'max_tokens': 500
                    }
                    
                    with st.spinner("Generating AI narratives for each subsection..."):
                        try:
                            narrative_results = report_mgr.generate_results_narratives(
                                assembled_result=st.session_state.assembled_results,
                                result_manager=result_mgr,
                                llm_config=llm_config
                            )
                            
                            st.session_state.narrative_results = narrative_results
                            st.session_state.llm_config_used = llm_config
                            
                            # Show success/failure summary
                            n_success = len(narrative_results.get('success', []))
                            n_failed = len(narrative_results.get('failed', []))
                            
                            st.success(f"✅ Generated {n_success} narratives successfully!")
                            
                            if n_failed > 0:
                                st.warning(f"⚠️ {n_failed} narratives failed to generate:")
                                for failure in narrative_results.get('failed', []):
                                    st.write(f"  • {failure}")
                        
                        except Exception as e:
                            st.error(f"Error generating narratives: {str(e)}")
                            return
                
                # Display narratives if generated
                if 'narrative_results' in st.session_state:
                    st.markdown("---")
                    st.subheader("📝 Generated Narratives")
                    
                    narrative_results = st.session_state.narrative_results
                    narratives = narrative_results.get('narratives', {})
                    
                    # Show individual narratives with edit capability
                    with st.expander("View & Edit Narratives", expanded=True):
                        st.markdown("You can review and edit each generated narrative before final download.")
                        
                        edited_narratives = {}
                        for placeholder, narrative in narratives.items():
                            st.markdown(f"**{placeholder}**")
                            edited_text = st.text_area(
                                label=f"Edit {placeholder}",
                                value=narrative,
                                height=150,
                                key=f"edit_{placeholder}",
                                label_visibility="collapsed"
                            )
                            edited_narratives[placeholder] = edited_text
                            st.markdown("---")
                        
                        # Update narratives with edits
                        if edited_narratives:
                            narrative_results['narratives'] = edited_narratives
                            
                            # Re-insert narratives with edits
                            narrative_results['latex_with_narratives'] = report_mgr._insert_narratives_into_template(
                                st.session_state.assembled_results['latex'],
                                edited_narratives
                            )
                            narrative_results['markdown_with_narratives'] = report_mgr._insert_narratives_into_template(
                                st.session_state.assembled_results['markdown'],
                                edited_narratives
                            )
                    
                    # Preview with narratives
                    st.markdown("**Final Results Section (with Narratives):**")

                    # PDF Compilation Section for Narratives (outside tabs)
                    import shutil
                    if shutil.which('pdflatex') is not None:
                        st.markdown("### 🔨 PDF Compilation (with Narratives)")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.info("Compile the LaTeX source with narratives to PDF. The compiled PDF will appear in the 'PDF Preview' tab below.")
                        with col2:
                            compile_narratives_clicked = st.button("🔨 Compile LaTeX to PDF", use_container_width=True, key="compile_pdf_narratives_btn_top")

                        if compile_narratives_clicked:
                            with st.spinner("Compiling LaTeX to PDF..."):
                                try:
                                    import subprocess
                                    import tempfile
                                    from pathlib import Path

                                    # Create a complete LaTeX document with necessary packages
                                    latex_full = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\usepackage{placeins}
\usepackage{adjustbox}

\title{Credit Risk Modeling Results}
\author{Research Team}
\date{\today}

\begin{document}
\maketitle

""" + narrative_results.get('latex_with_narratives', '') + r"""

\end{document}
"""

                                    # Create temporary directory for compilation
                                    with tempfile.TemporaryDirectory() as tmpdir:
                                        tmppath = Path(tmpdir)
                                        tex_file = tmppath / "results_complete.tex"

                                        # Write LaTeX file
                                        with open(tex_file, 'w', encoding='utf-8') as f:
                                            f.write(latex_full)

                                        # Copy all figures to temp directory
                                        figures_dir = Path("results/figures")
                                        if figures_dir.exists():
                                            for fig in figures_dir.glob("*.png"):
                                                import shutil as sh
                                                sh.copy(fig, tmppath / fig.name)

                                        # Run pdflatex (twice for references)
                                        for _ in range(2):
                                            result = subprocess.run(
                                                ['pdflatex', '-interaction=nonstopmode', 'results_complete.tex'],
                                                cwd=tmppath,
                                                capture_output=True,
                                                text=True,
                                                timeout=120
                                            )

                                        pdf_file = tmppath / "results_complete.pdf"

                                        if pdf_file.exists():
                                            # Read PDF and store in session state
                                            with open(pdf_file, 'rb') as f:
                                                pdf_data = f.read()

                                            st.session_state.compiled_pdf_narratives = pdf_data
                                            st.success("✅ PDF compiled successfully! Check the 'PDF Preview' tab below.")

                                            # Also save log for debugging
                                            log_file = tmppath / "results_complete.log"
                                            if log_file.exists():
                                                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                                    st.session_state.latex_log_narratives = f.read()
                                            
                                            # Rerun to refresh UI and show PDF
                                            st.rerun()
                                        else:
                                            st.error("❌ PDF compilation failed. Check LaTeX log below.")
                                            log_file = tmppath / "results_complete.log"
                                            if log_file.exists():
                                                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                                    log_content = f.read()
                                                with st.expander("View LaTeX Log"):
                                                    st.text(log_content)

                                except subprocess.TimeoutExpired:
                                    st.error("❌ PDF compilation timed out (>120s). Check your LaTeX code.")
                                except Exception as e:
                                    st.error(f"❌ Error during compilation: {str(e)}")

                        st.markdown("---")

                    final_tabs = st.tabs(["📝 Markdown with Narratives", "📜 LaTeX with Narratives", "📄 PDF Preview"])
                    
                    with final_tabs[0]:
                        with st.expander("View Full Markdown", expanded=False):
                            st.markdown(narrative_results.get('markdown_with_narratives', ''))
                    
                    with final_tabs[1]:
                        with st.expander("View LaTeX Code", expanded=False):
                            st.code(narrative_results.get('latex_with_narratives', ''), language='latex')
                    
                    with final_tabs[2]:
                        st.markdown("**PDF Preview (with Narratives):**")

                        # Display PDF if available (or show instructions if not compiled yet)
                        if 'compiled_pdf_narratives' in st.session_state:
                            st.markdown("---")

                            # PDF viewer using iframe
                            import base64
                            base64_pdf = base64.b64encode(st.session_state.compiled_pdf_narratives).decode('utf-8')
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)

                            # Download button
                            st.download_button(
                                label="⬇️ Download PDF (Complete)",
                                data=st.session_state.compiled_pdf_narratives,
                                file_name=f"results_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                            # Show log option
                            if 'latex_log_narratives' in st.session_state:
                                with st.expander("📋 View Compilation Log"):
                                    st.text(st.session_state.latex_log_narratives)
                        else:
                            # Show instructions when no PDF has been compiled
                            import shutil
                            if shutil.which('pdflatex') is None:
                                st.warning("⚠️ pdflatex not found on your system. Please install a LaTeX distribution (e.g., MiKTeX, TeX Live) to compile PDFs.")
                                st.info("You can still download the .tex file from the 'LaTeX with Narratives' tab and compile it manually.")
                            else:
                                st.info("📝 No PDF compiled yet. Click the '🔨 Compile LaTeX to PDF' button above to generate the PDF.")
                                st.info("💡 Tip: The compilation may take 30-60 seconds on the first run while LaTeX packages are installed.")
                    
                    # Download buttons for complete version
                    st.markdown("**Download Final Version:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="⬇️ Download Markdown (Complete)",
                            data=narrative_results.get('markdown_with_narratives', ''),
                            file_name=f"results_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.download_button(
                            label="⬇️ Download LaTeX (Complete)",
                            data=narrative_results.get('latex_with_narratives', ''),
                            file_name=f"results_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                            mime="text/x-tex",
                            use_container_width=True
                        )
            
            else:
                st.info("💡 Enable 'AI Commentary' above to add interpretive narratives to your Results section.")


    def _generate_latex_from_ai(self, csv_path: str, png_path: str, api_key: str) -> Optional[str]:
        """
        Use the provided API key to call OpenAI and produce a LaTeX
        'Results' section that includes the benchmark table and the AUC PNG.
        Returns the path to the generated .tex file or None on failure.
        """
        

    def _ai_summarise_shap_reliability(self, api_key: str, dataset: str, stab_df: Optional[pd.DataFrame], sanity_ratio: Optional[float], n_trials: int, bg_size: int) -> Optional[str]:
        """
        Use OpenAI to generate a short (2-4 sentence) concise summary of Global SHAP
        findings and an explanation of the Reliability Analysis (rank stability & sanity ratio).
        Returns the generated plain-text string or None on failure.
        """
        if openai is None:
            st.warning("OpenAI client library not available. Install the 'openai' package.")
            return None

        if not api_key:
            st.warning("No OpenAI API key provided.")
            return None

        # Prepare a small structured prompt with top features and reliability metrics
        try:
            # Build top-features text
            top_text = ""
            if stab_df is not None and not stab_df.empty:
                try:
                    # Prefer columns: feature, abs_mean_shap, avg_rank, std_rank
                    cols = stab_df.columns.tolist()
                    rows = stab_df.head(8)
                    lines = []
                    for _, r in rows.iterrows():
                        feat = str(r.get("feature", "<feature>"))
                        abs_mean = r.get("abs_mean", r.get("abs_mean_shap", r.get("abs_mean_shap", None)))
                        avg_rank = r.get("avg_rank", None)
                        std_rank = r.get("std_rank", None)
                        parts = [f"{feat}"]
                        if pd.notna(abs_mean):
                            parts.append(f"abs_mean={float(abs_mean):.4f}")
                        if pd.notna(avg_rank):
                            parts.append(f"avg_rank={float(avg_rank):.2f}")
                        if pd.notna(std_rank):
                            parts.append(f"std_rank={float(std_rank):.2f}")
                        lines.append(" (".join(parts) + ")" if False else ", ".join(parts))
                    top_text = "\n".join(lines)
                except Exception:
                    top_text = "(Top features unavailable)"
            else:
                top_text = "(No SHAP rank table available)"

            ratio_text = f"Sanity ratio: {sanity_ratio:.3f}" if sanity_ratio is not None and pd.notna(sanity_ratio) else "Sanity ratio: N/A"

            prompt = (
                "You are an assistant that writes concise, scientific summaries for machine-learning explainability outputs. "
                "Given Global SHAP rank/stability information and a reliability metric, produce a short 2-4 sentence paragraph that: (1) summarises the most important features and their relative influence, and (2) explains the reliability analysis (what the sanity ratio means and how avg_rank/std_rank reflect stability) and gives one actionable recommendation (e.g., increase trials, increase background size, or treat results cautiously). "
                "Keep language non-technical but precise and suitable for a figure caption or brief report blurb."
                "\n\nDATA:\n"
                f"Dataset: {dataset}\n"
                f"Top features (one per line):\n{top_text}\n"
                f"{ratio_text}\n"
                f"Stable-SHAP trials: {n_trials}, background size: {bg_size}\n"
            )

            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant that writes concise scientific summaries."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=180,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            st.warning(f"AI summarisation failed: {e}")
            return None

    def _generate_latex_from_ai(self, csv_path: str, png_path: str, api_key: str) -> Optional[str]:
        """
        Use the provided API key to call OpenAI and produce a LaTeX
        'Results' section that includes the benchmark table and the AUC PNG.
        Returns the path to the generated .tex file or None on failure.
        """
        if openai is None:
            st.warning("The 'openai' package is not installed. Install it to enable AI report generation.")
            return None

        # reading CSV for LaTeX generation
        try:
            df = pd.read_csv(csv_path)
            # CSV loaded successfully
        except Exception as e:
            st.warning(f"Could not read CSV for AI report generation: {e}")
            return None

        # Convert table to LaTeX string
        # converting table to LaTeX
        try:
            # Create a copy of the dataframe and escape underscores in string columns
            df_latex = df.copy()
            for col in df_latex.columns:
                if df_latex[col].dtype == 'object':  # string columns
                    df_latex[col] = df_latex[col].astype(str).str.replace('_', '\\_', regex=False)

            table_tex = df_latex.to_latex(index=False, caption="Benchmark Results", label="tab:benchmark", float_format="{:.4f}".format, escape=False)
        except Exception:
            # Fallback: simple tabular conversion with manual escaping
            df_latex = df.copy()
            for col in df_latex.columns:
                if df_latex[col].dtype == 'object':
                    df_latex[col] = df_latex[col].astype(str).str.replace('_', '\\_', regex=False)
            table_tex = df_latex.to_latex(index=False, escape=False)

        # Wrap the tabular environment with \resizebox{\textwidth}{!}{% ... }
        try:
            # Find the tabular block and wrap it so captions/labels remain outside
            tabular_match = re.search(r"(\\begin\{tabular\}.*?\\end\{tabular\})", table_tex, flags=re.DOTALL)
            if tabular_match:
                tabular_block = tabular_match.group(1)
                wrapped = "\\resizebox{\\textwidth}{!}{%\n" + tabular_block + "\n}"
                table_tex = table_tex.replace(tabular_block, wrapped)
        except Exception:
            # If wrapping fails, continue with unwrapped table_tex
            pass

        # Build LaTeX skeleton
        # Use a LaTeX-relative figures/ path for inclusion
        figure_basename = os.path.basename(png_path) if png_path else "model_comparison_AUC.png"
        # Use POSIX-style forward slash in LaTeX path regardless of OS
        figure_filename = f"figures/{figure_basename}"
        # figure filename for LaTeX: {figure_filename}

        # Default single-figure block (well-indented)
        figure_block = rf"""
\begin{{figure}}[ht]
    \centering
    \includegraphics[width=0.9\linewidth]{{{figure_filename}}}
    \caption{{Model comparison (AUC).}}
    \label{{fig:auc}}
\end{{figure}}
"""

        # Detect all model_comparison PNGs and build a 2x4 grid figure block if possible
        try:
            # Prefer figures folder next to the CSV (e.g., results/figures)
            csv_parent = Path(csv_path).parent
            candidate_dirs = [csv_parent / "figures", Path(__file__).parent / "results" / "figures"]
            figures_dir = None
            for d in candidate_dirs:
                if d.exists():
                    figures_dir = d
                    break
            if figures_dir is None:
                figures_dir = candidate_dirs[0]

            # Desired metric order for consistent layout
            # Exclude PCC and PG from the image grid (they remain present in the CSV/table)
            metrics_order = ["AUC", "F1", "Recall", "BS", "KS", "H"]
            imgs = []
            for m in metrics_order:
                fname = f"model_comparison_{m}.png"
                p = figures_dir / fname
                if p.exists():
                    imgs.append(f"figures/{fname}")

            # Fallback: if no images found by the ordered list, try globbing any model_comparison_*.png
            if not imgs:
                for p in sorted(figures_dir.glob("model_comparison_*.png")):
                    imgs.append(f"figures/{p.name}")

            # Ensure we have up to 6 slots (2 rows × 3 cols); pad with empty strings if necessary
            imgs = imgs[:6]
            while len(imgs) < 6:
                imgs.append("")

            # Build 2x3 tabular latex block (2 rows, 3 columns)
            row1 = imgs[0:3]
            row2 = imgs[3:6]

            def img_tex(path):
                if not path:
                    return ""
                return f"\\includegraphics[width=0.32\\textwidth]{{{path}}}"

            def build_row_tex(row):
                # create tex for non-empty cells; return None for an all-empty row
                cells = [img_tex(p) for p in row if p]
                if not cells:
                    return None
                return " &\n    ".join(cells)

            row_texts = [build_row_tex(r) for r in (row1, row2)]
            # Keep only non-empty rows to avoid stray blank rows like " &\n &\n"
            included_rows = [r for r in row_texts if r]

            if included_rows:
                rows_joined = " \\\\\n    ".join(included_rows)
                # overwrite figure_block with multi-image layout (2x3) using only included rows
                figure_block = rf"""
\begin{{figure}}[ht]
    \centering
    \begin{{tabular}}{{ccc}}
        {rows_joined}
    \end{{tabular}}
    \caption{{Model comparison across metrics.}}
    \label{{fig:model_comparisons}}
    \label{{fig:auc}}
\end{{figure}}
"""
            else:
                # No images found; keep the default single-image figure_block
                pass
        except Exception as e:
            st.warning(f"Could not build multi-image figure block: {e}")
            # keep default single-figure 'figure_block' defined above

        # Two-phase LLM pipeline: (1) summarise model families from code, (2) generate LaTeX Results
        try:
            client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            st.warning(f"Could not initialise OpenAI client: {e}")
            return None

        # Phase 1: summarise model families from models.py (best-effort)
        try:
            model_context = self._summarise_models_from_code(df, client)
        except Exception as e:
            st.warning(f"Model summarisation step failed: {e}")
            model_context = ""

        # Phase 2: build the Results-section prompt that uses the model-context
        try:
            user_instructions = (
                "Produce a LaTeX-formatted \\section{Results} of about 180-250 words. "
                "Start with an overview of overall performance across model families. "
                "Summarise key findings from the benchmark table, focusing on AUC as the primary metric and commenting on F1/Recall differences when relevant. "
                "When first mentioning each major model family (logistic regression, regularised logistic regression, random forests, bagged CART, AdaBoost / Boost-DT, gradient boosting, KNN, neural nets, XGBoost, LightGBM, etc.), "
                "use the MODEL-FAMILY DESCRIPTIONS to add a short clause (e.g. 'random forests (ensembles of bagged decision trees)'). "
                "Reference the figure strictly as \\ref{fig:auc}. End with implications for credit-risk modelling (which families are most promising and why), consistent with the numerical results. "
                "Use proper LaTeX formatting and escape underscores as \\_ in all model names (e.g. lr_reg -> lr\\_reg)."
            )

            model_context_block = model_context if model_context else "(No model-family context was available.)"

            user_msg = (
                user_instructions
                + "\n\nBENCHMARK TABLE (CSV format):\n"
                + df.to_string()
                + "\n\nMODEL-FAMILY DESCRIPTIONS:\n"
                + model_context_block
                + "\n\nNote: the main AUC figure is labelled 'fig:auc'."
            )

            messages = [
                {"role": "system", "content": "You are an assistant that writes LaTeX for academic papers. Be concise, numerically faithful to the data, and use proper LaTeX commands."},
                {"role": "user", "content": user_msg},
            ]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=800,
                temperature=0.2,
            )
            ai_text = response.choices[0].message.content
        except Exception as e:
            st.warning(f"AI generation failed: {e}")
            st.exception(e)
            return None

        # --- NEW: optionally generate per-dataset AI SHAP reliability summaries
        dataset_summaries_tex = ""
        try:
            # Collect datasets from global_shap_dfs (these are the ones that actually had SHAP run)
            datasets = []
            try:
                shap_cache = st.session_state.get("global_shap_dfs", {})
                if shap_cache:
                    datasets = list(shap_cache.keys())
                # Fallback to benchmark CSV if no SHAP cache
                elif "Dataset" in df.columns:
                    datasets = list(pd.unique(df["Dataset"]))
            except Exception:
                datasets = []

            # Limit to a reasonable number to avoid excessive API calls
            MAX_SUMMARIES = 6
            summary_calls = 0

            def _latex_escape(s: str) -> str:
                if s is None:
                    return ""
                esc = str(s)
                # minimal escaping for LaTeX special chars likely to appear
                for a, b in [("\\", "\\textbackslash{}"), ("$", "\\$"), ("%", "\\%"), ("_", "\\_"), ("&", "\\&"), ("#", "\\#"), ("{", "\\{"), ("}", "\\}")]:
                    esc = esc.replace(a, b)
                return esc

            if datasets and api_key:
                for ds_name in datasets:
                    if summary_calls >= MAX_SUMMARIES:
                        break
                    # Try to use cached reliability results if available; else use global_shap_dfs
                    stab_df = None
                    sanity_ratio = None
                    try:
                        stab_df = st.session_state.get("reliability_results", {}).get(ds_name)
                    except Exception:
                        stab_df = None
                    try:
                        sanity_ratio = st.session_state.get("reliability_ratios", {}).get(ds_name)
                    except Exception:
                        sanity_ratio = None

                    # If no stab_df, try global_shap_dfs (may lack ranks)
                    if stab_df is None:
                        try:
                            shap_cache = st.session_state.get("global_shap_dfs", {})
                            shap_df = shap_cache.get(ds_name)
                            # Provide a small fallback table (top features by abs_mean if available)
                            if shap_df is not None and not shap_df.empty:
                                stab_df = shap_df.head(8)
                        except Exception:
                            shap_df = None

                    try:
                        summary_calls += 1
                        # Use default values for n_trials and n_bg if not defined
                        ai_sum = self._ai_summarise_shap_reliability(api_key, ds_name, stab_df, sanity_ratio, 10, 200)
                    except Exception:
                        ai_sum = None

                    if ai_sum:
                        ds_label = _latex_escape(ds_name)
                        ai_escaped = _latex_escape(ai_sum)
                        dataset_summaries_tex += f"\\subsubsection*{{Global SHAP — {ds_label}}}\n{ai_escaped}\n\n"
                        
                        # Add bar and dot images in two columns
                        try:
                            bench_df = st.session_state.get("benchmark_results_df")
                            if bench_df is not None and not bench_df.empty:
                                # Find the benchmark model for this dataset
                                ds_rows = bench_df[bench_df["Dataset"] == ds_name]
                                if not ds_rows.empty:
                                    model_name = ds_rows.iloc[0]["Benchmark Model"]
                                    safe_ds = str(ds_name).replace(' ', '_').replace('.csv', '')
                                    safe_model = str(model_name).replace(' ', '_').replace('/', '_')
                                    
                                    bar_fig = f"figures/shap_{safe_ds}_{safe_model}_bar.png"
                                    dot_fig = f"figures/shap_{safe_ds}_{safe_model}_dot.png"
                                    
                                    # Check if files exist
                                    bar_path = Path(__file__).parent / "results" / bar_fig
                                    dot_path = Path(__file__).parent / "results" / dot_fig
                                    
                                    # Always add figures regardless of file existence (LaTeX will handle missing files)
                                    dataset_summaries_tex += "\\begin{figure}[H]\n"
                                    dataset_summaries_tex += "\\centering\n"
                                    dataset_summaries_tex += "\\begin{tabular}{cc}\n"
                                    dataset_summaries_tex += f"\\includegraphics[width=0.48\\textwidth]{{{bar_fig}}} &\n"
                                    dataset_summaries_tex += f"\\includegraphics[width=0.48\\textwidth]{{{dot_fig}}} \\\\\n"
                                    dataset_summaries_tex += "Bar Plot & Summary Plot (Dot) \\\\\n"
                                    dataset_summaries_tex += "\\end{tabular}\n"
                                    dataset_summaries_tex += f"\\caption{{Global SHAP plots for {ds_label} using {_latex_escape(model_name)}}}\n"
                                    dataset_summaries_tex += f"\\label{{fig:shap_{safe_ds}_{safe_model}}}\n"
                                    dataset_summaries_tex += "\\end{figure}\n\n"
                        except Exception as e:
                            # Non-fatal; continue without figures
                            st.warning(f"Could not add SHAP images for {ds_name}: {e}")
        except Exception:
            # Non-fatal; continue without per-dataset summaries
            dataset_summaries_tex = ""

        # Define a single LaTeX escape helper function to use throughout
        def _latex_escape(s: str) -> str:
            if s is None:
                return ""
            esc = str(s)
            # Important: escape backslash first, then other special chars
            for a, b in [("\\", "\\textbackslash "), ("$", "\\$"), ("%", "\\%"), ("_", "\\_"), ("&", "\\&"), ("#", "\\#"), ("{", "\\{"), ("}", "\\}")]:
                esc = esc.replace(a, b)
            return esc

        # If no per-dataset summaries were generated, attempt a single global summary fallback
        try:
            if (not dataset_summaries_tex.strip()) and api_key:
                # Compute an average sanity ratio if any are available
                ratios = []
                try:
                    ratios = [v for v in (st.session_state.get("reliability_ratios") or {}).values() if v is not None]
                except Exception:
                    ratios = []
                avg_ratio = float(np.mean(ratios)) if ratios else None

                overall_ai = None
                try:
                    overall_ai = self._ai_summarise_shap_reliability(api_key, "Overall", None, avg_ratio, int(st.session_state.get("rel_n_trials", 1)), int(st.session_state.get("rel_n_bg", 20)))
                except Exception:
                    overall_ai = None

                if overall_ai:
                    dataset_summaries_tex = f"\\subsection*{{SHAP Reliability Summary}}\\n{_latex_escape(overall_ai)}\\n\\n"
        except Exception:
            # safe fallback: leave empty
            pass

        # --- NEW: Build Local SHAP Analysis subsection if available ---
        local_shap_tex = ""
        try:
            analyses_dict = st.session_state.get("local_shap_analyses", {})
            if analyses_dict:
                # Include all analyses (sorted chronologically)
                sorted_ids = sorted(analyses_dict.keys(), key=lambda x: int(x.split('_')[-1]))
                
                local_shap_tex = "\\subsection*{Local SHAP Analyses}\n\n"
                
                for i, aid in enumerate(sorted_ids, 1):
                    rec = analyses_dict[aid]
                
                    ds_esc = _latex_escape(rec.get("dataset", "Unknown"))
                    model_esc = _latex_escape(rec.get("model_name", "Unknown"))
                    row_idx = rec.get("row_index", "?")
                    actual = rec.get("actual_target", "?")
                    pred_prob = rec.get("predicted_prob", 0.0)
                    commentary = rec.get("ai_commentary", "")
                    waterfall_png = rec.get("waterfall_png", None)
                    reliability_metrics = rec.get("reliability_metrics", None)
                    analysis_id = rec.get("analysis_id", aid)
                    
                    local_shap_tex += f"\\subsubsection*{{Analysis {i}: {analysis_id} (Row {row_idx})}}\n"
                    local_shap_tex += f"\\textbf{{Dataset:}} {ds_esc} \\\\\n"
                    local_shap_tex += f"\\textbf{{Model:}} {model_esc} \\\\\n"
                    local_shap_tex += f"\\textbf{{Actual Target:}} {actual} \\\\\n"
                    local_shap_tex += f"\\textbf{{Predicted Probability (for Class 1):}} {pred_prob:.4f}\n\n"
                    
                    # Add reliability metrics if available
                    if reliability_metrics:
                        score = reliability_metrics.get('reliability_score', 0)
                        bucket = reliability_metrics.get('reliability_bucket', 'Unknown')
                        sanity = reliability_metrics.get('sanity_ratio', 0)
                        er = reliability_metrics.get('explanatory_robustness', 0)
                        
                        local_shap_tex += "\\vspace{0.3cm}\n"
                        local_shap_tex += "\\textbf{Reliability Metrics:}\\\\\n"
                        local_shap_tex += f"\\textbf{{Score:}} {score:.3f} \\quad "
                        local_shap_tex += f"\\textbf{{Classification:}} {bucket} \\quad "
                        local_shap_tex += f"\\textbf{{Sanity Ratio:}} {sanity:.3f} \\quad "
                        local_shap_tex += f"\\textbf{{Explanatory Robustness:}} {er:.3f}\n\n"
                    
                    local_shap_tex += "\\vspace{0.3cm}\n"
                    
                    # Create two-column layout: Waterfall Plot on left, AI Explanation on right
                    if waterfall_png and commentary:
                        # Use forward slashes for LaTeX paths and strip results/ prefix
                        png_path = waterfall_png.replace("\\", "/")
                        # Remove results/ prefix if present for LaTeX relative path
                        if png_path.startswith("results/"):
                            png_path = png_path[8:]  # Remove "results/"
                        
                        commentary_esc = _latex_escape(commentary)
                        
                        local_shap_tex += "\\begin{figure}[H]\n"
                        local_shap_tex += "\\centering\n"
                        local_shap_tex += "\\begin{tabular}{p{0.48\\textwidth}p{0.48\\textwidth}}\n"
                        local_shap_tex += "\\textbf{Waterfall Plot} & \\textbf{AI Generated Explanation} \\\\\n"
                        local_shap_tex += "\\footnotesize How each feature pushes the prediction from the base value. & \\footnotesize A natural language summary of the prediction. \\\\\n"
                        local_shap_tex += f"\\includegraphics[width=0.48\\textwidth]{{{png_path}}} &\n"
                        local_shap_tex += f"\\begin{{minipage}}[t]{{0.48\\textwidth}}\n"
                        local_shap_tex += f"\\vspace{{0pt}}\n"
                        local_shap_tex += f"\\small {commentary_esc}\n"
                        local_shap_tex += f"\\end{{minipage}}\n"
                        local_shap_tex += "\\\\\n"
                        local_shap_tex += "\\end{tabular}\n"
                        local_shap_tex += f"\\caption{{Local SHAP analysis for {ds_esc}, row {row_idx}: Waterfall plot and AI-generated explanation.}}\n"
                        local_shap_tex += f"\\label{{fig:local_shap_{aid}_row{row_idx}}}\n"
                        local_shap_tex += "\\end{figure}\n\n"
                    elif waterfall_png:
                        # Only waterfall available
                        png_path = waterfall_png.replace("\\", "/")
                        if png_path.startswith("results/"):
                            png_path = png_path[8:]
                        local_shap_tex += f"""\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.75\\textwidth]{{{png_path}}}
    \\caption{{Waterfall plot for {ds_esc}, row {row_idx}.}}
    \\label{{fig:waterfall_{aid}_row{row_idx}}}
\\end{{figure}}

"""
                    elif commentary:
                        # Only commentary available
                        commentary_esc = _latex_escape(commentary)
                        local_shap_tex += f"\\textbf{{AI Generated Explanation:}}\n\n{commentary_esc}\n\n"
        except Exception as e:
            # Non-fatal; continue without local SHAP section
            local_shap_tex = ""
        
        # Compose final LaTeX: AI text + per-dataset summaries + local SHAP examples + table + figure
        final_tex = ai_text + "\n\n" + dataset_summaries_tex + local_shap_tex + "\n% Benchmark Results Table:\n" + table_tex + "\n" + figure_block

        # Save to report directory (place .tex in the same 'results' folder as the CSV)
        # saving LaTeX file
        try:
            report_dir = Path(csv_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)
            out_tex = report_dir / "results_section.tex"
            # saving to path
            with open(out_tex, "w", encoding="utf-8") as f:
                f.write(final_tex)
            # file saved successfully
            return str(out_tex)
        except Exception as e:
            st.warning(f"Could not save LaTeX file: {e}")
            st.exception(e)
            return None
            
    def run(self):
        """
        Run the main application logic and render the UI.
        """
        try:
            # Clear All Results button at the top
            col1, col2, col3 = st.columns([1, 5, 1])
            with col3:
                if st.button("🗑️ Clear Results", type="secondary", help="Delete all files in the results folder to start fresh", use_container_width=True):
                    from pathlib import Path
                    import shutil
                    results_path = Path("results")
                    if results_path.exists():
                        try:
                            # Delete all contents but keep the results folder
                            for item in results_path.iterdir():
                                if item.is_dir():
                                    shutil.rmtree(item)
                                else:
                                    item.unlink()
                            
                            # Recreate directory structure immediately
                            if self.result_mgr:
                                self.result_mgr.ensure_directory_structure()
                            
                            # Clear related session state to force fresh detection
                            keys_to_clear = [
                                'assembled_results', 'narratives_generated', 
                                'latex_results', 'markdown_results',
                                'results', 'benchmark_results'
                            ]
                            for key in keys_to_clear:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                            st.success("✅ All results cleared!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing results: {e}")
                    else:
                        st.info("No results to clear.")
            
            # Legacy "Export All Results" button - HIDDEN (replaced by Step 7)
            # One-click helper to export all results to files was here
            # Now handled by Step 7 "Generate Results Section" in Report Generation tab
            clicked_export = False

            # Section header for Data Preparation
            st.markdown("## 📊 Data Preparation")
            st.markdown("Upload datasets, compute feature importance, profile data, select features, and configure preprocessing.")
            st.markdown("---")

            # --- Step 1: Datasets ---
            self._render_step_1_dataset_selection()
            self._display_step_1_results()
            self._render_step_1_3_ydata_profiles()
            st.markdown("---")

            # --- NEW: Step 1.25 — Paper-Style Feature Importance ---
            # --- Step 1.25 — Paper-Style Feature Importance (RF & L1-LR) ---
            with st.expander("📊 Step 1.2: Compute Feature Importance (Optional)", expanded=False):
                have_data = bool(st.session_state.get("selected_datasets")) and bool(st.session_state.get("uploaded_files_map"))
                target = st.session_state.get("target_column", "target")

                if not have_data:
                    st.info("Upload datasets in Step 1 (and set target) to compute feature importance.")
                else:
                    # Build a signature of (dataset order, file bytes hash, target)
                    ds_names = [n for n in st.session_state.selected_datasets if n in st.session_state.uploaded_files_map]
                    sig_items, files_to_run = [], []
                    for name in ds_names:
                        fobj = st.session_state.uploaded_files_map[name]
                        try: fobj.seek(0)
                        except Exception: pass
                        sig_items.append((name, _bytesig_of_upload(fobj)))
                        files_to_run.append(fobj)
                    current_signature = (tuple(sig_items), target)

                    # Detect input change; mark as stale but DO NOT recompute
                    if st.session_state.fi_signature is not None and st.session_state.fi_signature != current_signature:
                        st.session_state.fi_stale = True

                    # Button: only this triggers computation
                    if st.button("Compute Feature Importance (per paper)", key="btn_fi_compute"):
                        try:
                            # Create progress indicators
                            progress_placeholder = st.empty()
                            status_placeholder = st.empty()

                            with progress_placeholder:
                                with st.spinner("🔄 Computing feature importance..."):
                                    status_placeholder.info(f"Processing {len(files_to_run)} dataset(s)... This may take a minute.")
                                    fi_results = compute_feature_importance_for_files(files_to_run, target=target)

                            # Clear progress indicators
                            progress_placeholder.empty()
                            status_placeholder.empty()

                            # Save results to disk using ResultManager
                            if self.result_mgr:
                                for ds_name, payload in fi_results.items():
                                    try:
                                        saved_paths = self.result_mgr.save_feature_importance(
                                            dataset_name=ds_name,
                                            rf_df=payload['rf'],
                                            lr_df=payload['lr'],
                                            merged_df=payload['merged'],
                                            metadata=payload.get('meta')
                                        )
                                        st.session_state.setdefault('feature_importance_paths', {})[ds_name] = saved_paths
                                    except Exception as save_err:
                                        st.warning(f"⚠️ Could not save feature importance for {ds_name}: {save_err}")

                            # Store results in session state
                            st.session_state.fi_results_cache = fi_results
                            st.session_state.fi_signature = current_signature
                            st.session_state.fi_stale = False

                            st.success(f"✅ Feature importance computed and saved for {len(fi_results)} dataset(s)!")
                            st.rerun()  # Rerun to show results immediately
                        except Exception as e:
                            st.error(f"❌ Failed to compute feature importance: {e}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())

                    # Show stale notice if inputs changed since last compute
                    if st.session_state.fi_stale:
                        st.warning("Inputs changed since last compute. Results below are from the previous run. Press the button to refresh.")

                    # Display cached results (persist across any UI change)
                    if st.session_state.fi_results_cache:
                        for ds_name, payload in st.session_state.fi_results_cache.items():
                            st.markdown(f"#### Dataset: `{ds_name}` — Top 20 (merged RF/LR)")
                            meta = payload.get("meta", {})
                            st.caption(
                                f"Rows: {meta.get('n_rows')}, Columns: {meta.get('n_cols')}, "
                                f"Kept after missing-drop: {len(meta.get('kept_columns_after_missing_drop', []))}"
                            )
                            st.dataframe(payload["merged"].head(20))

                            # 🚫 Do NOT use expanders inside an expander.
                            # ✅ Use tabs instead:
                            t1, t2 = st.tabs([
                                f"RandomForest importance (full) — {ds_name}",
                                f"LogisticRegression L1 |coef| (full) — {ds_name}",
                            ])
                            with t1:
                                st.dataframe(payload["rf"])
                            with t2:
                                st.dataframe(payload["lr"])
                    else:
                        st.info("No feature-importance results yet. Click the button to compute.")


            # after the Step 1.25 expander block
            self._render_step_1_4_feature_selector()
            st.markdown("---")

            # Step 1.5: Preprocessing Options (before tabs)
            if st.session_state.get("selected_datasets"):
                with st.expander("⚙️ Step 1.5: Preprocessing Options", expanded=False):
                    self._render_step_1_5_preprocessing_options()
                    self._display_step_1_5_results()
                st.markdown("---")

            # Section divider before main workflow tabs
            st.markdown("## 🔬 Analysis & Reporting")
            st.markdown("Choose between running experiments (**Research Lab**) or generating reports (**Report Generation**).")
            st.markdown("---")

            # Create tabs for Research Lab and Report Generation (always visible)
            tab_research, tab_report = st.tabs(["🔬 Research Lab", "📝 Report Generation"])

            with tab_research:
                # --- Steps 2, 3, 4, 5 (Conditional) ---
                if st.session_state.get("selected_datasets"):

                    with st.expander("🎯 Step 2: Select Target Variable & Models", expanded=False):
                        self._render_step_2_model_selection()
                        self._display_step_2_results()
                    st.markdown("---")

                    with st.expander("🔬 Step 3: Run Experiments", expanded=False):
                        if st.session_state.get("target_column") and st.session_state.get("selected_models"):
                            self._render_step_3_run_experiment()
                            self._display_step_3_results()
                        else:
                            st.info("Complete Step 2 (select target and models) to run the experiment.")

                    with st.expander("📊 Step 4: Benchmark Analysis", expanded=False):
                        self._render_step_4_benchmark_analysis()
                        self._display_step_4_results()

                    st.markdown("---")
                    # Step 5: No expander to prevent collapse on widget interaction
                    if st.session_state.get("benchmark_results_df") is not None:
                        self._render_step_5_shap_analysis()
                    else:
                        st.header("🔍 Step 5: Global SHAP Analysis")
                        st.info("Run Step 4 to identify benchmark models before running Global SHAP.")

                    # --- Step 5.5: Reliability Test (separate) ---
                    st.markdown("---")
                    if st.session_state.get("benchmark_results_df") is not None or st.session_state.get("selected_datasets"):
                        self._render_step_5_5_reliability_test()
                    else:
                        st.header("✅ Step 6: SHAP Reliability Tests")
                        st.info("Run Step 3/4 to produce benchmark models (or upload datasets) before running reliability tests.")

                    # --- Step 6: Local SHAP Analysis ---
                    st.markdown("---")
                    if st.session_state.get("benchmark_results_df") is not None:
                        self._render_step_6_local_analysis()
                    else:
                        # Still show the header, but with an info box
                        st.header("🔎 Step 7: Local SHAP Analysis")
                        st.info("Run Step 4 to identify benchmark models before running Local SHAP.")
                    
                    # --- Step 7.5: Batch Reliability Computation ---
                    # Only show if reliability features are enabled
                    if st.session_state.get("enable_reliability_toggle", False):
                        st.markdown("---")
                        if st.session_state.get("benchmark_results_df") is not None:
                            self._render_step_7_5_batch_reliability()
                        else:
                            st.header("📊 Step 7.5: Batch Reliability Computation")
                            st.info("Run Step 4 to identify benchmark models before computing batch reliability.")
                else:
                    st.info("Complete Step 1 (upload datasets) to proceed with analysis workflow.")

            # --- Tab 2: Report Generation ---
            with tab_report:
                self._render_step_7_report_generation()

            # Fire the expander-opening script after all sections are rendered
            fire_expand_all_if_pending()
        except Exception as e:
            st.error(f"An error occurred in the application: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


def main() -> None:
    """
    Main function to instantiate and run the Streamlit app.
    """
    app = ExperimentSetupApp()
    app.run()

if __name__ == "__main__":
    main()

