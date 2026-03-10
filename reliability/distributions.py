"""
distributions.py
----------------
Fit and evaluate reliability distributions on time-to-failure data.

Supported distributions:
    - Weibull (2-parameter)
    - Exponential
    - Lognormal
    - Gumbel (extreme value)
    - Pareto (heavy tail)

Each fitter returns:
    - MLE parameter estimates
    - 95% confidence intervals (bootstrap)
    - AIC / BIC for model comparison
    - Goodness-of-fit p-value (KS test)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma


# ---------------------------------------------------------------------
# Individual fitters
# ---------------------------------------------------------------------

def fit_weibull(data: np.ndarray, n_bootstrap: int = 500, seed: int = None):
    """
    Fit a 2-parameter Weibull distribution via MLE.

    scipy parameterization: shape=k, scale=lam
    We fix loc=0 (standard reliability assumption — no negative failure times).

    Parameters
    ----------
    data        : 1D array of positive failure times
    n_bootstrap : number of bootstrap resamples for CI estimation
    seed        : random seed

    Returns
    -------
    dict with keys:
        name, k, lam, aic, bic, ks_pvalue,
        k_ci, lam_ci  (each a [low, high] 95% interval)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    # MLE fit (loc fixed at 0)
    k, _, lam = stats.weibull_min.fit(data, floc=0)

    # Log-likelihood
    ll = np.sum(stats.weibull_min.logpdf(data, k, loc=0, scale=lam))

    # AIC / BIC (2 free parameters: k and lam)
    p = 2
    aic = 2 * p - 2 * ll
    bic = p * np.log(n) - 2 * ll

    # KS goodness-of-fit
    ks_stat, ks_p = stats.kstest(data, "weibull_min", args=(k, 0, lam))

    # Bootstrap CIs
    rng = np.random.default_rng(seed)
    boot_k, boot_lam = [], []
    for _ in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        try:
            bk, _, bl = stats.weibull_min.fit(resample, floc=0)
            boot_k.append(bk)
            boot_lam.append(bl)
        except Exception:
            pass

    k_ci = np.percentile(boot_k, [2.5, 97.5]).tolist()
    lam_ci = np.percentile(boot_lam, [2.5, 97.5]).tolist()

    return dict(
        name="Weibull",
        k=k,
        lam=lam,
        aic=aic,
        bic=bic,
        ks_pvalue=ks_p,
        k_ci=k_ci,
        lam_ci=lam_ci,
    )


def fit_lognormal(data: np.ndarray, n_bootstrap: int = 500, seed: int = None):
    """
    Fit a Lognormal distribution via MLE.

    Parameters: mu (log-scale mean), sigma (log-scale std).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    sigma, _, scale = stats.lognorm.fit(data, floc=0)
    mu = np.log(scale)

    ll = np.sum(stats.lognorm.logpdf(data, sigma, loc=0, scale=scale))
    p = 2
    aic = 2 * p - 2 * ll
    bic = p * np.log(n) - 2 * ll
    _, ks_p = stats.kstest(data, "lognorm", args=(sigma, 0, scale))

    rng = np.random.default_rng(seed)
    boot_mu, boot_sigma = [], []
    for _ in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        try:
            bs, _, bsc = stats.lognorm.fit(resample, floc=0)
            boot_mu.append(np.log(bsc))
            boot_sigma.append(bs)
        except Exception:
            pass

    return dict(
        name="Lognormal",
        mu=mu,
        sigma=sigma,
        aic=aic,
        bic=bic,
        ks_pvalue=ks_p,
        mu_ci=np.percentile(boot_mu, [2.5, 97.5]).tolist(),
        sigma_ci=np.percentile(boot_sigma, [2.5, 97.5]).tolist(),
    )


def fit_exponential(data: np.ndarray):
    """
    Fit an Exponential distribution (special case: Weibull with k=1).

    Single parameter: rate = 1/lam.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)

    _, scale = stats.expon.fit(data, floc=0)
    rate = 1.0 / scale

    ll = np.sum(stats.expon.logpdf(data, loc=0, scale=scale))
    p = 1
    aic = 2 * p - 2 * ll
    bic = p * np.log(n) - 2 * ll
    _, ks_p = stats.kstest(data, "expon", args=(0, scale))

    return dict(
        name="Exponential",
        rate=rate,
        lam=scale,
        aic=aic,
        bic=bic,
        ks_pvalue=ks_p,
    )


# ---------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------

def compare_distributions(data: np.ndarray, seed: int = 42) -> pd.DataFrame:
    """
    Fit all supported distributions and rank by AIC.

    Lower AIC = better fit (penalizes complexity).

    Parameters
    ----------
    data : 1D array of positive failure times
    seed : random seed for bootstrap

    Returns
    -------
    pd.DataFrame ranked by AIC with columns:
        distribution, aic, bic, ks_pvalue, params
    """
    results = []

    for fit_fn in [fit_weibull, fit_lognormal, fit_exponential]:
        try:
            if fit_fn == fit_exponential:
                r = fit_fn(data)
            else:
                r = fit_fn(data, seed=seed)
            results.append(r)
        except Exception as e:
            print(f"Warning: {fit_fn.__name__} failed: {e}")

    rows = []
    for r in results:
        rows.append({
            "distribution": r["name"],
            "aic": round(r["aic"], 2),
            "bic": round(r["bic"], 2),
            "ks_pvalue": round(r["ks_pvalue"], 4),
            "params": {k: v for k, v in r.items()
                       if k not in ("name", "aic", "bic", "ks_pvalue")},
        })

    df = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def weibull_survival(t: np.ndarray, k: float, lam: float) -> np.ndarray:
    """
    Weibull survival function: S(t) = P(T > t) = exp(-(t/lam)^k)
    """
    return np.exp(-((t / lam) ** k))


def weibull_b_life(k: float, lam: float, pct: float) -> float:
    """
    B-life: time at which pct% of units have failed.
    Inverts the Weibull CDF: t = lam * (-ln(1 - p))^(1/k)
    """
    p = pct / 100.0
    return lam * (-np.log(1 - p)) ** (1.0 / k)


def weibull_mttf(k: float, lam: float) -> float:
    """MTTF = lam * Gamma(1 + 1/k)"""
    return lam * gamma(1 + 1.0 / k)


# ---------------------------------------------------------------------
# Quick demo: recover true parameters from simulated data
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from simulate import weibull_sample, sparse_sample, mttf, b_life

    TRUE_K = 2.5
    TRUE_LAM = 1000.0

    print("=" * 60)
    print(f"True Weibull(k={TRUE_K}, lam={TRUE_LAM})")
    print(f"True MTTF    : {mttf(TRUE_K, TRUE_LAM):.1f} hrs")
    print(f"True B10 life: {b_life(TRUE_K, TRUE_LAM, 10):.1f} hrs")
    print("=" * 60)

    # Show how MLE degrades as n shrinks
    samples = sparse_sample(TRUE_K, TRUE_LAM, seed=42)

    print(f"\n{'n':>6} | {'k_est':>8} | {'k_95CI':>18} | {'lam_est':>9} | {'lam_95CI':>20}")
    print("-" * 70)

    for n, data in samples.items():
        r = fit_weibull(data, seed=42)
        k_ci = f"[{r['k_ci'][0]:.2f}, {r['k_ci'][1]:.2f}]"
        lam_ci = f"[{r['lam_ci'][0]:.0f}, {r['lam_ci'][1]:.0f}]"
        print(f"{n:>6} | {r['k']:>8.3f} | {k_ci:>18} | {r['lam']:>9.1f} | {lam_ci:>20}")

    # Distribution comparison on n=100
    print("\n--- Distribution comparison (n=100) ---")
    data_100 = samples[100]
    comparison = compare_distributions(data_100)
    print(comparison[["rank", "distribution", "aic", "bic", "ks_pvalue"]].to_string(index=False))
