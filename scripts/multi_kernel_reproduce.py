# -*- coding: utf-8 -*-
"""
Multi-kernel software reliability prediction reproduction script.

Implementation details:
1. Modified LSCV cross term consistent with the manuscript formula:
   sum_i [(1/n) sum_j Lambda_hat_{(-j),h}(x_i)] * Lambda_hat_h(x_i)
2. Removed the hard 1e9 validity threshold in LSCV.
   Only NaN/Inf are treated as invalid.
3. Normalizes Y only inside the optimization objective for numerical stability.
   Prediction still uses the original cumulative fault counts.
4. Uses 3-logit softmax parameterization for the four kernel weights.
5. Reads Excel datasets from a data directory such as data/OSS or data/CSS and
   writes compact reproduction output containing only predicted sequences and
   PMAE.
"""

import datetime
import hashlib
import argparse
import math
import os
import warnings
import traceback
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd
from numba import njit

try:
    from pyswarm import pso
except ImportError:
    pso = None

warnings.filterwarnings("ignore")

# PSO settings used in both profiles. Increase these for the final full run if needed.
DEFAULT_NUM_RESTARTS = 5
DEFAULT_SWARMSIZE = 50
DEFAULT_MAXITER = 50
DEFAULT_MINFUNC = 1e-6
DEFAULT_LSCV_GRID_SIZE = 100

# Weight refinement is kept as an option and disabled by default.
USE_WEIGHT_REFINE = False

INVALID_SCORE = 1e30


# ============================================================
# 1. Error metrics
# ============================================================
def mean_absolute_error(y_true, y_pred):
    n = len(y_true)
    return np.sum(np.abs(np.array(y_pred) - np.array(y_true))) / n


# ============================================================
# 2. Kernel functions, Python interface
# ============================================================
def Gauss_kernel(t, h):
    x = t / h
    return (1 / h) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)


def Triangularkernel(t, h):
    x = t / h
    return np.where(np.abs(x) <= 1, (1 - np.abs(x)) / h, 0)


def Biweight(t, h):
    x = t / h
    return np.where(np.abs(x) <= 1, (15 / 16) * ((1 - x ** 2) ** 2) / h, 0)


def Epanechnikovkernel(t, h):
    x = t / h
    return np.where(np.abs(x) <= 1, 0.75 * (1 - x ** 2) / h, 0)


def Uniform_kernel(t, h):
    x = t / h
    return np.where(np.abs(x) <= 1, 0.5 / h, 0)


def composite_kernel_4(t, h_list, w_list):
    return (
        w_list[0] * Gauss_kernel(t, h_list[0])
        + w_list[1] * Triangularkernel(t, h_list[1])
        + w_list[2] * Biweight(t, h_list[2])
        + w_list[3] * Epanechnikovkernel(t, h_list[3])
    )


kernelDictt = {"composite_kernel_4": composite_kernel_4}

ESTIMATOR_ID_MAP = {
    "NW_estimator": 0,
    "LL_estimator": 1,
}


# ============================================================
# 3. Python estimator interface, kept for compatibility
# ============================================================
def NW_estimator(X_train, y_train, x_test, h_list, w_list, kernel_func):
    X_train = np.asarray(X_train).reshape(-1, 1)
    y_train = np.asarray(y_train).reshape(-1, 1)
    x_test_array = np.atleast_1d(x_test).reshape(1, -1)

    t_diffs = x_test_array - X_train
    k_vals = kernel_func(t_diffs, h_list, w_list)

    numerator = np.sum(k_vals * y_train, axis=0)
    denominator = np.sum(k_vals, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        y_pred = numerator / denominator
    y_pred = np.nan_to_num(y_pred, nan=0.0)

    if np.isscalar(x_test):
        return y_pred.item()
    return y_pred


def LL_estimator(X_train, y_train, x_test, h_list, w_list, composite_kernel_func_arg):
    X_train = np.asarray(X_train).reshape(-1, 1)
    y_train = np.asarray(y_train).reshape(-1, 1)
    x_test_array = np.atleast_1d(x_test).reshape(1, -1)

    t_diffs = x_test_array - X_train
    k_vals = composite_kernel_func_arg(t_diffs, h_list, w_list)

    S1 = np.sum(k_vals * t_diffs, axis=0)
    S2 = np.sum(k_vals * (t_diffs ** 2), axis=0)
    w_contribs = k_vals * (S2 - t_diffs * S1)

    numerator = np.sum(w_contribs * y_train, axis=0)
    denominator = np.sum(w_contribs, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        y_pred = numerator / denominator
    y_pred = np.nan_to_num(y_pred, nan=0.0)

    if np.isscalar(x_test):
        return y_pred.item()
    return y_pred


# ============================================================
# 4. Numba accelerated multi-kernel prediction and modified LSCV
# ============================================================
@njit(cache=True, fastmath=True)
def _base_kernel_value(diff, h, kernel_id):
    if h <= 0.0:
        return 0.0

    x = diff / h
    ax = abs(x)

    # 0: Gaussian
    if kernel_id == 0:
        return (1.0 / h) * (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

    # 1: Triangular
    if kernel_id == 1:
        if ax <= 1.0:
            return (1.0 - ax) / h
        return 0.0

    # 2: Biweight
    if kernel_id == 2:
        if ax <= 1.0:
            tmp = 1.0 - x * x
            return (15.0 / 16.0) * (tmp * tmp) / h
        return 0.0

    # 3: Epanechnikov
    if kernel_id == 3:
        if ax <= 1.0:
            return 0.75 * (1.0 - x * x) / h
        return 0.0

    return 0.0


@njit(cache=True, fastmath=True)
def _composite_kernel_value(diff, h_arr, w_arr):
    total = 0.0
    for k in range(4):
        total += w_arr[k] * _base_kernel_value(diff, h_arr[k], k)
    return total


@njit(cache=True, fastmath=True)
def _nw_multi_predict_point(X, Y, x_test, h_arr, w_arr):
    numerator = 0.0
    denominator = 0.0

    for i in range(X.shape[0]):
        kval = _composite_kernel_value(x_test - X[i], h_arr, w_arr)
        numerator += kval * Y[i]
        denominator += kval

    if denominator == 0.0 or math.isnan(denominator) or math.isinf(denominator):
        return 0.0

    y = numerator / denominator
    if math.isnan(y) or math.isinf(y):
        return 0.0
    return y


@njit(cache=True, fastmath=True)
def _ll_multi_predict_point(X, Y, x_test, h_arr, w_arr):
    n = X.shape[0]
    S1 = 0.0
    S2 = 0.0

    for i in range(n):
        d = x_test - X[i]
        kval = _composite_kernel_value(d, h_arr, w_arr)
        S1 += kval * d
        S2 += kval * d * d

    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        d = x_test - X[i]
        kval = _composite_kernel_value(d, h_arr, w_arr)
        wc = kval * (S2 - d * S1)
        numerator += wc * Y[i]
        denominator += wc

    if denominator == 0.0 or math.isnan(denominator) or math.isinf(denominator):
        return 0.0

    y = numerator / denominator
    if math.isnan(y) or math.isinf(y):
        return 0.0
    return y


@njit(cache=True, fastmath=True)
def _multi_predict_point(X, Y, x_test, h_arr, w_arr, estimator_id):
    if estimator_id == 0:
        return _nw_multi_predict_point(X, Y, x_test, h_arr, w_arr)
    return _ll_multi_predict_point(X, Y, x_test, h_arr, w_arr)


@njit(cache=True, fastmath=True)
def _nw_multi_predict_point_loo(X, Y, x_test, h_arr, w_arr, leave_idx):
    numerator = 0.0
    denominator = 0.0

    for i in range(X.shape[0]):
        if i == leave_idx:
            continue
        kval = _composite_kernel_value(x_test - X[i], h_arr, w_arr)
        numerator += kval * Y[i]
        denominator += kval

    if denominator == 0.0 or math.isnan(denominator) or math.isinf(denominator):
        return 0.0

    y = numerator / denominator
    if math.isnan(y) or math.isinf(y):
        return 0.0
    return y


@njit(cache=True, fastmath=True)
def _ll_multi_predict_point_loo(X, Y, x_test, h_arr, w_arr, leave_idx):
    n = X.shape[0]
    S1 = 0.0
    S2 = 0.0

    for i in range(n):
        if i == leave_idx:
            continue
        d = x_test - X[i]
        kval = _composite_kernel_value(d, h_arr, w_arr)
        S1 += kval * d
        S2 += kval * d * d

    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        if i == leave_idx:
            continue
        d = x_test - X[i]
        kval = _composite_kernel_value(d, h_arr, w_arr)
        wc = kval * (S2 - d * S1)
        numerator += wc * Y[i]
        denominator += wc

    if denominator == 0.0 or math.isnan(denominator) or math.isinf(denominator):
        return 0.0

    y = numerator / denominator
    if math.isnan(y) or math.isinf(y):
        return 0.0
    return y


@njit(cache=True, fastmath=True)
def _multi_predict_point_loo(X, Y, x_test, h_arr, w_arr, estimator_id, leave_idx):
    if estimator_id == 0:
        return _nw_multi_predict_point_loo(X, Y, x_test, h_arr, w_arr, leave_idx)
    return _ll_multi_predict_point_loo(X, Y, x_test, h_arr, w_arr, leave_idx)


@njit(cache=True, fastmath=True)
def _calculate_lscv_integral_multi_numba(X, Y, h_arr, w_arr, estimator_id, num_grid):
    n = X.shape[0]
    if n == 0 or num_grid < 2:
        return INVALID_SCORE

    xmin = X[0]
    xmax = X[0]
    for i in range(1, n):
        if X[i] < xmin:
            xmin = X[i]
        if X[i] > xmax:
            xmax = X[i]

    if xmin == xmax:
        padding = 0.1
    else:
        padding = (xmax - xmin) * 0.1

    a = xmin - padding
    b = xmax + padding
    dx = (b - a) / (num_grid - 1)

    total = 0.0
    for g in range(num_grid):
        xg = a + dx * g
        y = _multi_predict_point(X, Y, xg, h_arr, w_arr, estimator_id)
        val = y * y

        if math.isnan(val) or math.isinf(val):
            return INVALID_SCORE

        if g == 0 or g == num_grid - 1:
            total += 0.5 * val
        else:
            total += val

    out = total * dx
    if math.isnan(out) or math.isinf(out):
        return INVALID_SCORE
    return out


@njit(cache=True, fastmath=True)
def _lscv_cross_term_multi_numba(X, Y, h_arr, w_arr, estimator_id):
    n = X.shape[0]
    if n == 0:
        return INVALID_SCORE

    total = 0.0
    for i in range(n):
        xi = X[i]
        full_pred = _multi_predict_point(X, Y, xi, h_arr, w_arr, estimator_id)

        loo_sum = 0.0
        for j in range(n):
            loo_pred = _multi_predict_point_loo(X, Y, xi, h_arr, w_arr, estimator_id, j)
            loo_sum += loo_pred

        loo_avg = loo_sum / n
        total += loo_avg * full_pred

    if math.isnan(total) or math.isinf(total):
        return INVALID_SCORE
    return total


@njit(cache=True, fastmath=True)
def _lscv_score_multi_numba(X, Y, h_arr, w_arr, estimator_id, num_grid):
    integral_term = _calculate_lscv_integral_multi_numba(X, Y, h_arr, w_arr, estimator_id, num_grid)
    cross_term = _lscv_cross_term_multi_numba(X, Y, h_arr, w_arr, estimator_id)

    # Important: do not reject a score just because it is large.
    # Large finite values can be valid for cumulative fault-count data.
    if math.isnan(integral_term) or math.isinf(integral_term):
        return INVALID_SCORE
    if math.isnan(cross_term) or math.isinf(cross_term):
        return INVALID_SCORE

    score = integral_term - 2.0 * cross_term
    if math.isnan(score) or math.isinf(score):
        return INVALID_SCORE
    return score


@njit(cache=True, fastmath=True)
def _predict_next_multi_numba(X, Y, h_arr, w_arr, estimator_id):
    i = X.shape[0]
    pred_1 = _multi_predict_point(X, Y, 1.0, h_arr, w_arr, estimator_id)
    pred_prev = _multi_predict_point(X, Y, i / (i + 1.0), h_arr, w_arr, estimator_id)
    return Y[i - 1] + (pred_1 - pred_prev)


# ============================================================
# 5. PSO optimization
# ============================================================
def bandwidth_bounds(n_train):
    h_max = 1.0
    h0 = 0.05
    k_neighbors = 3
    eps = 0.05

    if n_train <= 0:
        raise ValueError("n_train must be positive.")

    h_min = min(h_max - eps, max(h0, k_neighbors / n_train))
    return float(h_min), float(h_max)


def stable_seed(dataset_name, estimator_name, start_idx, restart_id, mode=""):
    text = f"{dataset_name}_{estimator_name}_{start_idx}_{restart_id}_{mode}"
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = np.sum(w)
    if s <= 0 or np.isnan(s) or np.isinf(s):
        return np.ones_like(w) / len(w)
    return w / s


def alpha_from_theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape[0] != 3:
        raise ValueError("theta must have length 3 for four-kernel softmax weights.")

    z = np.zeros(4, dtype=np.float64)
    z[:3] = theta
    z = z - np.max(z)
    e = np.exp(z)
    s = np.sum(e)

    if s <= 0 or np.isnan(s) or np.isinf(s):
        return np.ones(4, dtype=np.float64) / 4.0
    return e / s


def midpoint_equal_weights(num_kernels, h_min, h_max):
    h_mid = (h_min + h_max) / 2.0
    return [h_mid] * num_kernels, [1.0 / num_kernels] * num_kernels


def unpack_pso_result(pso_result):
    """
    Robustly unpack return values from different pso/pyswarm versions.

    Some environments return exactly (xopt, fopt), while others return
    additional diagnostic values such as iteration history. We only need
    the optimized parameter vector and its objective value.
    """
    if isinstance(pso_result, (tuple, list)):
        if len(pso_result) < 2:
            raise ValueError(f"Unexpected pso return length: {len(pso_result)}")
        return pso_result[0], pso_result[1], len(pso_result)

    # A rare case: some optimizers may return an object with x/fun fields.
    if hasattr(pso_result, "x") and hasattr(pso_result, "fun"):
        return pso_result.x, pso_result.fun, "OptimizeResult"

    raise TypeError(f"Unexpected pso return type: {type(pso_result)}")


def refine_weights_given_h_pso(
    X,
    Y_for_objective,
    fixed_h,
    estimator_id,
    lscv_grid_size=100,
    swarmsize=80,
    maxiter=100,
    minfunc=1e-6,
):
    if pso is None:
        raise ImportError("pyswarm is not installed. Please run: pip install pyswarm")

    X = np.asarray(X, dtype=np.float64)
    Y_for_objective = np.asarray(Y_for_objective, dtype=np.float64)
    fixed_h = np.asarray(fixed_h, dtype=np.float64)

    theta_lb = np.array([-8.0, -8.0, -8.0], dtype=float)
    theta_ub = np.array([8.0, 8.0, 8.0], dtype=float)

    def objective_theta(theta):
        w_obj = alpha_from_theta(theta).astype(np.float64)
        score = _lscv_score_multi_numba(
            X,
            Y_for_objective,
            fixed_h,
            w_obj,
            estimator_id,
            lscv_grid_size,
        )
        if np.isnan(score) or np.isinf(score):
            return INVALID_SCORE
        return float(score)

    pso_result = pso(
        objective_theta,
        theta_lb,
        theta_ub,
        swarmsize=swarmsize,
        maxiter=maxiter,
        minfunc=minfunc,
        debug=False,
    )
    theta_opt, score_opt, _pso_return_len = unpack_pso_result(pso_result)

    w_opt = alpha_from_theta(theta_opt).astype(np.float64)
    real_score = _lscv_score_multi_numba(
        X,
        Y_for_objective,
        fixed_h,
        w_opt,
        estimator_id,
        lscv_grid_size,
    )
    if np.isfinite(real_score):
        score_opt = float(real_score)
    else:
        score_opt = INVALID_SCORE

    return w_opt, float(score_opt)


def find_optimal_parameters_pso(
    X,
    Y,
    composite_kernel_func_arg,
    estimator_fn,
    dataset_name="unknown",
    estimator_name="unknown",
    start_idx=0,
    mode="Fixed",
    previous_h=None,
    previous_w=None,
    num_restarts=DEFAULT_NUM_RESTARTS,
    swarmsize=DEFAULT_SWARMSIZE,
    maxiter=DEFAULT_MAXITER,
    minfunc=DEFAULT_MINFUNC,
    lscv_grid_size=DEFAULT_LSCV_GRID_SIZE,
    use_weight_refine=USE_WEIGHT_REFINE,
):
    """
    PSO optimizer with detailed diagnostics for fallback.

    This version records and prints:
    - objective probe scores before PSO;
    - integral/cross/score values for probe parameters;
    - PSO exceptions instead of silently swallowing them;
    - objective call counts and invalid-score counts;
    - each restart's raw pso score and recomputed real score.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(X)
    num_kernels = 4
    estimator_id = ESTIMATOR_ID_MAP[estimator_name]

    # Normalize Y only for LSCV optimization. Prediction still uses raw Y.
    y_scale = max(float(np.max(np.abs(Y))), 1.0) if len(Y) > 0 else 1.0
    Y_obj = Y / y_scale

    h_min, h_max = bandwidth_bounds(n)

    h_bounds_min = [h_min] * num_kernels
    h_bounds_max = [h_max] * num_kernels
    theta_bounds_min = [-8.0] * 3
    theta_bounds_max = [8.0] * 3

    lb = np.array(h_bounds_min + theta_bounds_min, dtype=float)
    ub = np.array(h_bounds_max + theta_bounds_max, dtype=float)

    exception_log = []
    restart_details = []
    objective_stats = {
        "calls": 0,
        "finite": 0,
        "invalid": 0,
        "min_finite": np.inf,
        "max_finite": -np.inf,
        "first_invalid_examples": [],
    }

    def objective(params):
        objective_stats["calls"] += 1

        try:
            params = np.asarray(params, dtype=np.float64)
            h_list_obj = np.asarray(params[:num_kernels], dtype=np.float64)
            w_list_obj = alpha_from_theta(params[num_kernels:]).astype(np.float64)

            if np.any(h_list_obj < h_min - 1e-12) or np.any(h_list_obj > h_max + 1e-12):
                objective_stats["invalid"] += 1
                if len(objective_stats["first_invalid_examples"]) < 5:
                    objective_stats["first_invalid_examples"].append("h_out_of_bounds")
                return INVALID_SCORE

            score = _lscv_score_multi_numba(
                X,
                Y_obj,
                h_list_obj,
                w_list_obj,
                estimator_id,
                lscv_grid_size,
            )

            if np.isnan(score) or np.isinf(score):
                objective_stats["invalid"] += 1
                if len(objective_stats["first_invalid_examples"]) < 5:
                    objective_stats["first_invalid_examples"].append("score_nan_or_inf")
                return INVALID_SCORE

            score = float(score)
            if score >= INVALID_SCORE * 0.1:
                objective_stats["invalid"] += 1
                if len(objective_stats["first_invalid_examples"]) < 5:
                    objective_stats["first_invalid_examples"].append("score_near_invalid")
                return score

            objective_stats["finite"] += 1
            if score < objective_stats["min_finite"]:
                objective_stats["min_finite"] = score
            if score > objective_stats["max_finite"]:
                objective_stats["max_finite"] = score
            return score

        except Exception as e:
            objective_stats["invalid"] += 1
            msg = "objective_exception: " + repr(e)
            if len(objective_stats["first_invalid_examples"]) < 5:
                objective_stats["first_invalid_examples"].append(msg)
            if len(exception_log) < 20:
                exception_log.append(msg + "\n" + traceback.format_exc())
            return INVALID_SCORE

    if pso is None:
        raise ImportError("pyswarm is not installed. Please run: pip install pyswarm")

    best_h = None
    best_w = None
    best_score = np.inf
    restart_scores = []
    restart_seeds = []

    weight_refine_used = False
    weight_refine_improved = False
    weight_refine_score = None

    fallback_used = False
    fallback_type = "none"
    optimization_status = "success"

    if n < 4:
        exception_log.append(f"n_too_small_for_pso: n={n}")

    if n >= 4:
        for restart_id in range(num_restarts):
            seed = stable_seed(dataset_name, estimator_name, start_idx, restart_id, mode)
            restart_seeds.append(seed)
            np.random.seed(seed)

            h_candidate = None
            w_candidate = None
            raw_pso_score = INVALID_SCORE
            real_score = INVALID_SCORE
            score = INVALID_SCORE
            pso_exception = None
            pso_return_len = None

            try:
                pso_result = pso(
                    objective,
                    lb,
                    ub,
                    swarmsize=swarmsize,
                    maxiter=maxiter,
                    minfunc=minfunc,
                    debug=False,
                )
                params, raw_pso_score, pso_return_len = unpack_pso_result(pso_result)

                h_candidate = np.asarray(params[:num_kernels], dtype=np.float64)
                w_candidate = alpha_from_theta(params[num_kernels:]).astype(np.float64)

                real_score = _lscv_score_multi_numba(
                    X,
                    Y_obj,
                    h_candidate,
                    w_candidate,
                    estimator_id,
                    lscv_grid_size,
                )
                if np.isfinite(real_score):
                    score = float(real_score)
                else:
                    score = INVALID_SCORE

            except Exception as e:
                pso_exception = repr(e)
                tb = traceback.format_exc()
                exception_log.append(
                    f"pso_exception restart={restart_id}: {repr(e)}\n{tb}"
                )
                print(
                    f"[PSO ERROR] dataset={dataset_name}, estimator={estimator_name}, "
                    f"start_idx={start_idx}, restart={restart_id}, error={repr(e)}"
                )

            restart_scores.append(float(score))
            restart_details.append({
                "restart_id": restart_id,
                "seed": seed,
                "raw_pso_score": float(raw_pso_score) if np.isfinite(raw_pso_score) else str(raw_pso_score),
                "real_score": float(real_score) if np.isfinite(real_score) else str(real_score),
                "accepted_score": float(score),
                "h_candidate": None if h_candidate is None else h_candidate.tolist(),
                "w_candidate": None if w_candidate is None else w_candidate.tolist(),
                "exception": pso_exception,
                "pso_return_len": pso_return_len,
            })

            if h_candidate is not None and np.isfinite(score) and score < best_score and score < INVALID_SCORE * 0.1:
                best_score = float(score)
                best_h = h_candidate.astype(float)
                best_w = w_candidate.astype(float)

    if best_h is not None and best_w is not None and np.isfinite(best_score) and use_weight_refine:
        try:
            refine_seed = stable_seed(dataset_name, estimator_name, start_idx, 9999, mode + "_WeightRefine")
            np.random.seed(refine_seed)
            refined_w, refined_score = refine_weights_given_h_pso(
                X,
                Y_obj,
                best_h,
                estimator_id,
                lscv_grid_size=lscv_grid_size,
                swarmsize=max(swarmsize, 80),
                maxiter=max(maxiter, 100),
                minfunc=minfunc,
            )
            weight_refine_used = True
            weight_refine_score = float(refined_score)
            if np.isfinite(refined_score) and refined_score < best_score:
                best_w = refined_w.astype(float)
                best_score = float(refined_score)
                weight_refine_improved = True
        except Exception as e:
            weight_refine_used = False
            weight_refine_improved = False
            weight_refine_score = None
            exception_log.append("weight_refine_exception: " + repr(e) + "\n" + traceback.format_exc())

    if best_h is None or best_w is None or (not np.isfinite(best_score)):
        fallback_used = True
        optimization_status = "fallback"

        if previous_h is not None and previous_w is not None:
            best_h = np.asarray(previous_h, dtype=np.float64)
            best_w = normalize_weights(previous_w).astype(np.float64)
            fallback_type = "previous_valid_parameters"
        else:
            best_h, best_w = midpoint_equal_weights(num_kernels, h_min, h_max)
            best_h = np.asarray(best_h, dtype=np.float64)
            best_w = np.asarray(best_w, dtype=np.float64)
            fallback_type = "midpoint_equal_weights"

        try:
            best_score = _lscv_score_multi_numba(
                X,
                Y_obj,
                best_h,
                best_w,
                estimator_id,
                lscv_grid_size,
            )
            if not np.isfinite(best_score):
                best_score = INVALID_SCORE
        except Exception as e:
            best_score = INVALID_SCORE
            exception_log.append("fallback_score_exception: " + repr(e) + "\n" + traceback.format_exc())

    # Convert inf diagnostics to readable values.
    objective_min_finite = objective_stats["min_finite"]
    objective_max_finite = objective_stats["max_finite"]
    if not np.isfinite(objective_min_finite):
        objective_min_finite = None
    if not np.isfinite(objective_max_finite):
        objective_max_finite = None

    meta = {
        "H_Min": h_min,
        "H_Max": h_max,
        "Num_Restarts": num_restarts,
        "Swarm_Size": swarmsize,
        "Max_Iter": maxiter,
        "MinFunc": minfunc,
        "LSCV_Grid_Size": lscv_grid_size,
        "Invalid_Score": INVALID_SCORE,
        "Objective_Uses_Normalized_Y": True,
        "Y_Scale": y_scale,
        "Weight_Parameterization": "3-logit-softmax",
        "Theta_Bounds": "[-8, 8] for theta1-theta3; theta4 fixed at 0",
        "Weight_Refine_Used": weight_refine_used,
        "Weight_Refine_Improved": weight_refine_improved,
        "Weight_Refine_Score": weight_refine_score,
        "Restart_Scores": restart_scores,
        "Restart_Seeds": restart_seeds,
        "Restart_Details": restart_details,
        "Objective_Call_Count": objective_stats["calls"],
        "Objective_Finite_Count": objective_stats["finite"],
        "Objective_Invalid_Count": objective_stats["invalid"],
        "Objective_Min_Finite": objective_min_finite,
        "Objective_Max_Finite": objective_max_finite,
        "Objective_First_Invalid_Examples": objective_stats["first_invalid_examples"],
        "Exception_Log": exception_log,
        "Fallback_Used": fallback_used,
        "Fallback_Type": fallback_type,
        "Optimization_Status": optimization_status,
    }

    return best_h.tolist(), best_w.tolist(), float(best_score), meta

# ============================================================
# 6. Experiment routines
# ============================================================
def make_result_row(dataset_name, kernel_name, estimator_name, mode, actual_vals, preds,
                    h_hist, w_hist, score_hist, meta_hist, target_ratio, actual_start_ratio):
    return {
        "Dataset": dataset_name,
        "Estimator": estimator_name,
        "Bandwidth_Mode": mode,
        "Target_Start_Ratio": float(target_ratio),
        "Actual_Start_Ratio": float(actual_start_ratio),
        "PMAE": mean_absolute_error(actual_vals, preds),
        "Predicted_Y_Sequence": ",".join(map(str, preds)),
    }


def normalized_x(n):
    return np.arange(1, n + 1, dtype=np.float64) / float(n)


def process_single_stage(args):
    """
    Process exactly one experiment mode.

    Previous version:
        one task = Fixed + Dynamic together

    New version:
        one task = only Fixed or only Dynamic

    This makes Fixed and Dynamic independently scheduled by multiprocessing.Pool.
    Note: Dynamic still needs the same initial PSO optimization as Fixed, so when
    both modes are enabled, the initial optimization is computed once in the
    Fixed task and once in the Dynamic task. This increases total optimization
    work slightly, but improves scheduling and lets Fixed results finish without
    waiting for Dynamic.
    """
    (
        dataset_name,
        DS_original,
        kernel_name,
        composite_kernel_func,
        estimator_name,
        estimator_fn,
        start_idx,
        target_ratio,
        task_mode,
        pso_num_restarts,
        pso_swarmsize,
        pso_maxiter,
        pso_minfunc,
        pso_lscv_grid_size,
        use_weight_refine,
    ) = args

    if task_mode not in ("Fixed", "Dynamic"):
        raise ValueError(f"Unknown task_mode: {task_mode}")

    estimator_id = ESTIMATOR_ID_MAP[estimator_name]
    DS_arr = np.asarray(DS_original, dtype=np.float64)
    number = len(DS_arr)
    predictionLimit = number

    base_Y = DS_arr[:start_idx].copy()
    base_X = normalized_x(start_idx)

    # Initial optimization is required by both Fixed and Dynamic.
    # Keep mode="Fixed_initial" to preserve the same deterministic seed and
    # reproduce the original initial parameter selection.
    opt_h, opt_w, opt_score, opt_meta = find_optimal_parameters_pso(
        base_X,
        base_Y,
        composite_kernel_func,
        estimator_fn,
        dataset_name=dataset_name,
        estimator_name=estimator_name,
        start_idx=start_idx,
        mode="Fixed_initial",
        previous_h=None,
        previous_w=None,
        num_restarts=pso_num_restarts,
        swarmsize=pso_swarmsize,
        maxiter=pso_maxiter,
        minfunc=pso_minfunc,
        lscv_grid_size=pso_lscv_grid_size,
        use_weight_refine=use_weight_refine,
    )

    actual_vals = DS_arr[start_idx:predictionLimit]

    if task_mode == "Fixed":
        opt_h_arr = np.asarray(opt_h, dtype=np.float64)
        opt_w_arr = np.asarray(opt_w, dtype=np.float64)

        curr_Y_fixed = np.empty(predictionLimit, dtype=np.float64)
        curr_Y_fixed[:start_idx] = base_Y
        preds_fixed = np.empty(predictionLimit - start_idx, dtype=np.float64)

        i = start_idx
        while i < predictionLimit:
            curr_X = normalized_x(i)
            pred_next = _predict_next_multi_numba(
                curr_X,
                curr_Y_fixed[:i],
                opt_h_arr,
                opt_w_arr,
                estimator_id,
            )
            curr_Y_fixed[i] = pred_next
            preds_fixed[i - start_idx] = pred_next
            i += 1

        res_fixed = make_result_row(
            dataset_name=dataset_name,
            kernel_name=kernel_name,
            estimator_name=estimator_name,
            mode="Fixed",
            actual_vals=actual_vals,
            preds=preds_fixed.tolist(),
            h_hist=[opt_h],
            w_hist=[opt_w],
            score_hist=[opt_score],
            meta_hist=[opt_meta],
            target_ratio=target_ratio,
            actual_start_ratio=start_idx / number,
        )
        return [res_fixed]

    # task_mode == "Dynamic"
    curr_Y_dyn = np.empty(predictionLimit, dtype=np.float64)
    curr_Y_dyn[:start_idx] = base_Y
    preds_dyn = np.empty(predictionLimit - start_idx, dtype=np.float64)

    dyn_h_hist = [opt_h]
    dyn_w_hist = [opt_w]
    dyn_score_hist = [opt_score]
    dyn_meta_hist = [opt_meta]
    h_dyn, w_dyn = opt_h, opt_w
    h_dyn_arr = np.asarray(h_dyn, dtype=np.float64)
    w_dyn_arr = np.asarray(w_dyn, dtype=np.float64)

    i = start_idx
    first = True
    while i < predictionLimit:
        curr_X = normalized_x(i)
        if first:
            first = False
        else:
            h_dyn, w_dyn, s_dyn, meta_dyn = find_optimal_parameters_pso(
                curr_X,
                curr_Y_dyn[:i],
                composite_kernel_func,
                estimator_fn,
                dataset_name=dataset_name,
                estimator_name=estimator_name,
                start_idx=i,
                mode="Dynamic_step",
                previous_h=h_dyn,
                previous_w=w_dyn,
                num_restarts=pso_num_restarts,
                swarmsize=pso_swarmsize,
                maxiter=pso_maxiter,
                minfunc=pso_minfunc,
                lscv_grid_size=pso_lscv_grid_size,
                use_weight_refine=use_weight_refine,
            )
            dyn_h_hist.append(h_dyn)
            dyn_w_hist.append(w_dyn)
            dyn_score_hist.append(s_dyn)
            dyn_meta_hist.append(meta_dyn)
            h_dyn_arr = np.asarray(h_dyn, dtype=np.float64)
            w_dyn_arr = np.asarray(w_dyn, dtype=np.float64)

        pred_next = _predict_next_multi_numba(
            curr_X,
            curr_Y_dyn[:i],
            h_dyn_arr,
            w_dyn_arr,
            estimator_id,
        )
        curr_Y_dyn[i] = pred_next
        preds_dyn[i - start_idx] = pred_next
        i += 1

    res_dyn = make_result_row(
        dataset_name=dataset_name,
        kernel_name=kernel_name,
        estimator_name=estimator_name,
        mode="Dynamic",
        actual_vals=actual_vals,
        preds=preds_dyn.tolist(),
        h_hist=dyn_h_hist,
        w_hist=dyn_w_hist,
        score_hist=dyn_score_hist,
        meta_hist=dyn_meta_hist,
        target_ratio=target_ratio,
        actual_start_ratio=start_idx / number,
    )

    return [res_dyn]

OUTPUT_COLUMNS = [
    "Dataset",
    "Target_Start_Ratio",
    "Actual_Start_Ratio",
    "Predicted_Y_Sequence",
    "PMAE",
]


def load_datasets_from_excel_folder(data_folder_name="data"):
    data_dir = os.path.abspath(data_folder_name)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    excel_files = []
    for fname in os.listdir(data_dir):
        lower = fname.lower()
        if lower.startswith("~$"):
            continue
        if lower.endswith((".xlsx", ".xlsm", ".xls")):
            excel_files.append(fname)
    excel_files = sorted(excel_files)

    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in: {data_dir}")

    datasets = {}
    for fname in excel_files:
        file_path = os.path.join(data_dir, fname)
        dataset_name = os.path.splitext(fname)[0]

        try:
            df = pd.read_excel(file_path, header=None)
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel file: {file_path}, error={repr(e)}")

        df = df.dropna(how="all")
        non_empty_cols = [col for col in df.columns if not df[col].dropna().empty]
        if len(non_empty_cols) == 0:
            raise ValueError(f"{fname} has no non-empty columns.")

        if len(non_empty_cols) == 1:
            y_col = non_empty_cols[0]
            y_values = pd.to_numeric(df[y_col], errors="coerce").dropna().astype(float).to_numpy()
        else:
            x_col, y_col = non_empty_cols[:2]
            work = df[[x_col, y_col]].copy()
            work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
            work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
            work = work.dropna(subset=[x_col, y_col])
            work = work.sort_values(by=x_col).reset_index(drop=True)
            y_values = work[y_col].astype(float).to_numpy()

        if len(y_values) < 3:
            raise ValueError(f"{fname} has too few valid numeric points; at least 3 are required.")

        datasets[dataset_name] = y_values.tolist()
        print(f"Loaded dataset: {dataset_name}, points={len(y_values)}, file={fname}")

    print(f"\nLoaded {len(datasets)} Excel dataset(s).")
    return datasets


def parse_start_ratios(value):
    ratios = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ratio = float(item)
        if ratio <= 0.0 or ratio >= 1.0:
            raise argparse.ArgumentTypeError("start ratios must be between 0 and 1")
        ratios.append(ratio)
    if not ratios:
        raise argparse.ArgumentTypeError("at least one start ratio is required")
    return ratios


def parse_args():
    default_processes = min(4, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(description="Reproduce compact multi-kernel results.")
    parser.add_argument("--data-dir", default="data/OSS", help="Directory containing Excel datasets. Default: data/OSS.")
    parser.add_argument("--output", default=None, help="Output .xlsx path.")
    parser.add_argument("--processes", type=int, default=default_processes)
    parser.add_argument("--start-ratios", type=parse_start_ratios, default=parse_start_ratios("0.2,0.5,0.8"))
    parser.add_argument("--fixed-only", action="store_true", help="Run Fixed mode only.")
    parser.add_argument("--num-restarts", type=int, default=DEFAULT_NUM_RESTARTS)
    parser.add_argument("--swarmsize", type=int, default=DEFAULT_SWARMSIZE)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--minfunc", type=float, default=DEFAULT_MINFUNC)
    parser.add_argument("--lscv-grid-size", type=int, default=DEFAULT_LSCV_GRID_SIZE)
    parser.add_argument("--use-weight-refine", action="store_true")
    return parser.parse_args()


def warm_up_numba():
    try:
        warm_X = np.array([0.5, 1.0], dtype=np.float64)
        warm_Y = np.array([1.0, 2.0], dtype=np.float64)
        warm_h = np.array([0.95, 0.95, 0.95, 0.95], dtype=np.float64)
        warm_w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        _lscv_score_multi_numba(warm_X, warm_Y, warm_h, warm_w, 0, 10)
        _lscv_score_multi_numba(warm_X, warm_Y, warm_h, warm_w, 1, 10)
        _predict_next_multi_numba(warm_X, warm_Y, warm_h, warm_w, 0)
        _predict_next_multi_numba(warm_X, warm_Y, warm_h, warm_w, 1)
    except Exception:
        pass


def build_reproduction_tasks(datasets, estimators_dict, kernel_funcs_dict, args):
    tasks = []
    run_dynamic = not args.fixed_only

    for kernel_name, kernel_func in kernel_funcs_dict.items():
        for estimator_name, estimator_fn in estimators_dict.items():
            for dataset_name, dataset_values in datasets.items():
                number = len(dataset_values)
                seen = set()
                for ratio in args.start_ratios:
                    start_idx = math.ceil(number * ratio)
                    if start_idx in seen:
                        continue
                    seen.add(start_idx)
                    if not (2 <= start_idx < number):
                        print(f"Skipping {dataset_name} at ratio {ratio}: invalid start index {start_idx} for n={number}.")
                        continue

                    common = (
                        dataset_name,
                        dataset_values,
                        kernel_name,
                        kernel_func,
                        estimator_name,
                        estimator_fn,
                        start_idx,
                        ratio,
                    )
                    pso_params = (
                        args.num_restarts,
                        args.swarmsize,
                        args.maxiter,
                        args.minfunc,
                        args.lscv_grid_size,
                        args.use_weight_refine,
                    )
                    tasks.append(common + ("Fixed",) + pso_params)
                    if run_dynamic:
                        tasks.append(common + ("Dynamic",) + pso_params)

    return tasks


def run_reproduction_tasks(tasks, processes):
    results = []
    if processes <= 1:
        for i, task in enumerate(tasks, start=1):
            results.extend(process_single_stage(task))
            if i % 10 == 0 or i == len(tasks):
                print(f"Progress: {i}/{len(tasks)} tasks completed")
        return results

    with Pool(processes=processes) as pool:
        for i, result_list in enumerate(pool.imap_unordered(process_single_stage, tasks), start=1):
            results.extend(result_list)
            if i % 10 == 0 or i == len(tasks):
                print(f"Progress: {i}/{len(tasks)} tasks completed")
    return results


def write_compact_workbook(results, output_path):
    sheet_map = {
        ("NW_estimator", "Fixed"): "NW_Fixed",
        ("NW_estimator", "Dynamic"): "NW_Dynamic",
        ("LL_estimator", "Fixed"): "LL_Fixed",
        ("LL_estimator", "Dynamic"): "LL_Dynamic",
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for (estimator, mode), sheet_name in sheet_map.items():
            rows = [
                {col: result[col] for col in OUTPUT_COLUMNS}
                for result in results
                if result["Estimator"] == estimator and result["Bandwidth_Mode"] == mode
            ]
            df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
            if not df.empty:
                df = df.sort_values(["Dataset", "Target_Start_Ratio"]).reset_index(drop=True)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    if pso is None:
        raise ImportError("pyswarm is required for multi_kernel_reproduce.py. Install it with: python -m pip install pyswarm")

    args = parse_args()
    if args.processes < 1:
        raise ValueError("--processes must be at least 1")
    if args.num_restarts < 1:
        raise ValueError("--num-restarts must be at least 1")
    if args.swarmsize < 1:
        raise ValueError("--swarmsize must be at least 1")
    if args.maxiter < 1:
        raise ValueError("--maxiter must be at least 1")
    if args.lscv_grid_size < 2:
        raise ValueError("--lscv-grid-size must be at least 2")

    datasets = load_datasets_from_excel_folder(args.data_dir)
    estimators_to_run = {"NW_estimator": NW_estimator, "LL_estimator": LL_estimator}

    output_path = args.output
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"multi_kernel_results_{timestamp}.xlsx"
    output_path = os.path.abspath(output_path)

    tasks = build_reproduction_tasks(datasets, estimators_to_run, kernelDictt, args)
    if not tasks:
        raise RuntimeError("No valid prediction tasks were generated.")

    print(f"Working directory: {os.getcwd()}")
    print(f"Output file: {output_path}")
    print(f"Tasks: {len(tasks)}")
    print(f"Processes: {args.processes}")

    warm_up_numba()
    results = run_reproduction_tasks(tasks, args.processes)
    write_compact_workbook(results, output_path)
    print(f"\nDone. Results saved to: {output_path}")


# ============================================================
if __name__ == "__main__":
    freeze_support()
    main()
