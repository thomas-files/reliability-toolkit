"""
tail_analysis.py
----------------
Bayesian and frequentist methods for tail characterization
under sparse data conditions (n = 5-30 observations).

Key idea:
    MLE confidence intervals explode at small n.
    Bayesian priors regularize the estimates using domain knowledge,
    keeping tail predictions meaningful even with 5-10 observations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

# PyMC is optional — graceful error if not installed
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


# ---------------------------------------------------------------------
# Bayesian Weibull model
# ---------------------------------------------------------------------

def bayesian_weibull(
    data: np.ndarray,
    k_prior_mu: float = 1.0,
    k_prior_sigma: float = 0.4,
    lam_prior_mu: float = None,
    lam_prior_sigma: float = 0.5,
    draws: int = 2000,
    tune: int = 1000,
    seed: int = 42,
):
    """
    Fit a Bayesian Weibull model using MCMC (PyMC).

    Priors (both LogNormal to enforce positivity):
        k   ~ LogNormal(k_prior_mu, k_prior_sigma)
            default centers near e^1 ≈ 2.7, reasonable for wear-out
        lam ~ LogNormal(ln(data_mean), lam_prior_sigma)
            default centers on the sample mean as a weak guess

    Parameters
    ----------
    data              : 1D array of positive failure times
    k_prior_mu        : log-scale mean for k prior
    k_prior_sigma     : log-scale std for k prior (larger = more uncertain)
    lam_prior_mu      : log-scale mean for lam prior (default: log(mean(data)))
    lam_prior_sigma   : log-scale std for lam prior
    draws             : MCMC posterior samples
    tune              : MCMC tuning steps
    seed              : random seed

    Returns
    -------
    az.InferenceData object (contains posterior samples for k and lam)
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC not installed. Run: pip install pymc")

    data = np.asarray(data, dtype=float)

    if lam_prior_mu is None:
        lam_prior_mu = np.log(np.mean(data))

    with pm.Model() as model:
        # Priors
        k = pm.LogNormal("k", mu=k_prior_mu, sigma=k_prior_sigma)
        lam = pm.LogNormal("lam", mu=lam_prior_mu, sigma=lam_prior_sigma)

        # Likelihood — Weibull parameterized as in scipy (alpha=k, beta=lam)
        obs = pm.Weibull("obs", alpha=k, beta=lam, observed=data)

        # Sample posterior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                draws=draws,
                tune=tune,
                random_seed=seed,
                progressbar=True,
                return_inferencedata=True,
            )

    return trace


def posterior_summary(trace) -> pd.DataFrame:
    """
    Extract posterior mean and 94% credible interval for k and lam.

    Parameters
    ----------
    trace : az.InferenceData from bayesian_weibull()

    Returns
    -------
    pd.DataFrame with columns: param, mean, hdi_3%, hdi_97%
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC not installed.")

    summary = az.summary(trace, var_names=["k", "lam"], hdi_prob=0.94)
    return summary[["mean", "hdi_3%", "hdi_97%"]]


def posterior_tail_probability(
    trace,
    t: float,
) -> dict:
    """
    Compute posterior predictive P(T > t) — survival probability at time t.

    For each MCMC sample (k_i, lam_i), compute S(t) = exp(-(t/lam_i)^k_i).
    This gives a full distribution over S(t), not just a point estimate.

    Parameters
    ----------
    trace : az.InferenceData
    t     : time threshold

    Returns
    -------
    dict with keys: mean, median, hdi_low, hdi_high, samples
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC not installed.")

    k_samples = trace.posterior["k"].values.flatten()
    lam_samples = trace.posterior["lam"].values.flatten()

    survival = np.exp(-((t / lam_samples) ** k_samples))

    hdi = az.hdi(survival, hdi_prob=0.94)

    return {
        "t": t,
        "mean": float(np.mean(survival)),
        "median": float(np.median(survival)),
        "hdi_low": float(hdi[0]),
        "hdi_high": float(hdi[1]),
        "samples": survival,
    }


# ---------------------------------------------------------------------
# Prior sensitivity analysis
# ---------------------------------------------------------------------

def prior_sensitivity(
    data: np.ndarray,
    prior_scenarios: list = None,
    draws: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Test how sensitive posterior estimates are to prior choice.

    Runs the Bayesian model under several different priors and compares
    the resulting k and lam posteriors. Large sensitivity = priors matter
    a lot (usually because data is sparse). Small sensitivity = data
    dominates (usually because n is large enough).

    Parameters
    ----------
    data             : 1D array of failure times
    prior_scenarios  : list of dicts, each with keys:
                         label, k_prior_mu, k_prior_sigma,
                         lam_prior_mu, lam_prior_sigma
    draws            : MCMC samples per scenario
    seed             : random seed

    Returns
    -------
    pd.DataFrame with one row per scenario showing posterior summaries
    """
    if prior_scenarios is None:
        lam_guess = np.log(np.mean(data))
        prior_scenarios = [
            dict(label="Weak prior",
                 k_prior_mu=0.5, k_prior_sigma=1.0,
                 lam_prior_mu=lam_guess, lam_prior_sigma=1.0),
            dict(label="Moderate prior (default)",
                 k_prior_mu=1.0, k_prior_sigma=0.4,
                 lam_prior_mu=lam_guess, lam_prior_sigma=0.5),
            dict(label="Strong prior (k~2.5)",
                 k_prior_mu=np.log(2.5), k_prior_sigma=0.2,
                 lam_prior_mu=lam_guess, lam_prior_sigma=0.3),
        ]

    rows = []
    for scenario in prior_scenarios:
        label = scenario.pop("label")
        trace = bayesian_weibull(data, draws=draws, seed=seed, **scenario)
        k_post = trace.posterior["k"].values.flatten()
        lam_post = trace.posterior["lam"].values.flatten()
        rows.append({
            "scenario": label,
            "k_mean": round(float(np.mean(k_post)), 3),
            "k_std": round(float(np.std(k_post)), 3),
            "lam_mean": round(float(np.mean(lam_post)), 1),
            "lam_std": round(float(np.std(lam_post)), 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_tail_uncertainty(
    data: np.ndarray,
    trace,
    true_k: float = None,
    true_lam: float = None,
    t_max: float = None,
):
    """
    Plot survival function uncertainty from posterior samples.

    Shows:
        - Posterior predictive band (94% HDI across MCMC samples)
        - Posterior mean survival curve
        - True survival curve (if true params provided)
        - Observed failure times as rug plot

    Parameters
    ----------
    data     : observed failure times
    trace    : az.InferenceData from bayesian_weibull()
    true_k   : true shape (for ground truth comparison)
    true_lam : true scale (for ground truth comparison)
    t_max    : x-axis limit (default: 2 * max(data))
    """
    if t_max is None:
        t_max = 2 * np.max(data)

    t = np.linspace(0.01, t_max, 300)

    k_samples = trace.posterior["k"].values.flatten()
    lam_samples = trace.posterior["lam"].values.flatten()

    # Survival curve for each posterior sample
    survival_curves = np.array([
        np.exp(-((t / lam_i) ** k_i))
        for k_i, lam_i in zip(k_samples[:500], lam_samples[:500])
    ])

    mean_curve = survival_curves.mean(axis=0)
    lower = np.percentile(survival_curves, 3, axis=0)
    upper = np.percentile(survival_curves, 97, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Survival function with uncertainty band ---
    ax = axes[0]
    ax.fill_between(t, lower, upper, alpha=0.25, color="steelblue",
                    label="94% posterior band")
    ax.plot(t, mean_curve, color="steelblue", lw=2, label="Posterior mean")

    if true_k is not None and true_lam is not None:
        true_curve = np.exp(-((t / true_lam) ** true_k))
        ax.plot(t, true_curve, color="crimson", lw=2,
                linestyle="--", label=f"True (k={true_k}, λ={true_lam})")

    ax.plot(np.sort(data), np.linspace(1, 0, len(data)),
            "o", color="black", ms=4, alpha=0.6, label="Observed failures")

    ax.set_xlabel("Time")
    ax.set_ylabel("S(t) = P(T > t)")
    ax.set_title(f"Survival Function Uncertainty  (n={len(data)})")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # --- Right: Posterior distributions for k and lam ---
    ax2 = axes[1]
    k_post = trace.posterior["k"].values.flatten()
    lam_post = trace.posterior["lam"].values.flatten()

    ax2.hist(k_post, bins=50, density=True, alpha=0.6,
             color="steelblue", label=f"k  (mean={k_post.mean():.2f})")
    ax2.axvline(k_post.mean(), color="steelblue", lw=2)

    if true_k is not None:
        ax2.axvline(true_k, color="crimson", lw=2,
                    linestyle="--", label=f"True k={true_k}")

    ax2.set_xlabel("k (shape parameter)")
    ax2.set_ylabel("Density")
    ax2.set_title("Posterior Distribution of k")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("tail_uncertainty.png", dpi=150, bbox_inches="tight")
    print("Plot saved: tail_uncertainty.png")
    plt.show()


# ---------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    if not PYMC_AVAILABLE:
        print("PyMC not installed. Run: pip install pymc arviz")
        print("Showing frequentist comparison only.\n")

        from distributions import fit_weibull
        from simulate import sparse_sample

        TRUE_K, TRUE_LAM = 2.5, 1000.0
        samples = sparse_sample(TRUE_K, TRUE_LAM, seed=42)

        print(f"{'n':>6} | {'k_mle':>8} | {'k_95CI (bootstrap)':>22}")
        print("-" * 45)
        for n, data in samples.items():
            r = fit_weibull(data, seed=42)
            ci = f"[{r['k_ci'][0]:.2f}, {r['k_ci'][1]:.2f}]"
            print(f"{n:>6} | {r['k']:>8.3f} | {ci:>22}")

    else:
        from simulate import weibull_sample

        TRUE_K, TRUE_LAM = 2.5, 1000.0
        print(f"Fitting Bayesian Weibull on n=10 (true k={TRUE_K}, lam={TRUE_LAM})")

        data = weibull_sample(10, TRUE_K, TRUE_LAM, seed=42)
        trace = bayesian_weibull(data)

        print("\nPosterior summary:")
        print(posterior_summary(trace))

        result = posterior_tail_probability(trace, t=500)
        print(f"\nP(T > 500 hrs):  mean={result['mean']:.3f}  "
              f"94% HDI=[{result['hdi_low']:.3f}, {result['hdi_high']:.3f}]")

        plot_tail_uncertainty(data, trace, true_k=TRUE_K, true_lam=TRUE_LAM)
