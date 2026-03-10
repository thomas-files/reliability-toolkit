"""
simulate.py
-----------
Generate synthetic time-to-failure datasets for testing and demonstration.

Useful for:
    - Benchmarking estimators under known ground truth
    - Demonstrating sparse-data failure modes
    - Stress-testing the automated report pipeline
"""

import numpy as np
import pandas as pd


def weibull_sample(n: int, k: float, lam: float, seed: int = None) -> np.ndarray:
    """
    Draw n failure times from a Weibull(k, lam) distribution.

    Uses inverse CDF sampling:
        T = lam * (-ln(U))^(1/k),  U ~ Uniform(0, 1)

    Parameters
    ----------
    n   : number of samples
    k   : shape parameter (k<1 infant mortality, k=1 random, k>1 wear-out)
    lam : scale parameter (characteristic life — time at which ~63.2% have failed)
    seed: random seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n,) with failure times
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)
    return lam * (-np.log(u)) ** (1.0 / k)


def censored_sample(
    n: int,
    k: float,
    lam: float,
    observation_time: float,
    seed: int = None,
) -> pd.DataFrame:
    """
    Simulate right-censored field data.

    Units are observed for `observation_time` hours. Units that haven't
    failed by then are censored — we record their time as observation_time
    and mark them as not failed.

    This mirrors real deployment data where you pull a report at time T
    and some units are still running.

    Parameters
    ----------
    n                : number of units in the field
    k                : Weibull shape
    lam              : Weibull scale
    observation_time : cutoff time (e.g. 8760 = 1 year in hours)
    seed             : random seed

    Returns
    -------
    pd.DataFrame with columns:
        - time     : observed time (failure time or censoring time)
        - failed   : True if unit actually failed, False if censored
    """
    times = weibull_sample(n, k, lam, seed=seed)
    failed = times <= observation_time
    observed_time = np.where(failed, times, observation_time)

    return pd.DataFrame({"time": observed_time, "failed": failed})


def sparse_sample(
    k: float,
    lam: float,
    n_values: list = None,
    seed: int = None,
) -> dict:
    """
    Generate multiple subsamples at different n values from the same distribution.

    Used to demonstrate how MLE estimates degrade as sample size shrinks.
    Each subsample is drawn independently (not nested).

    Parameters
    ----------
    k        : true Weibull shape
    lam      : true Weibull scale
    n_values : list of sample sizes, e.g. [5, 10, 15, 30, 100]
    seed     : base random seed (each n gets seed + i for reproducibility)

    Returns
    -------
    dict mapping n -> np.ndarray of failure times
    """
    if n_values is None:
        n_values = [5, 10, 15, 30, 100]

    return {
        n: weibull_sample(n, k, lam, seed=seed + i if seed is not None else None)
        for i, n in enumerate(n_values)
    }


def mttf(k: float, lam: float) -> float:
    """
    True mean time to failure for Weibull(k, lam).

    MTTF = lam * Gamma(1 + 1/k)

    Parameters
    ----------
    k   : shape
    lam : scale

    Returns
    -------
    float
    """
    from scipy.special import gamma
    return lam * gamma(1 + 1.0 / k)


def b_life(k: float, lam: float, pct: float) -> float:
    """
    B-life: time by which `pct` percent of units have failed.

    B10 life = time at which 10% have failed (common reliability metric).
    Derived by inverting the Weibull CDF:
        t = lam * (-ln(1 - pct/100))^(1/k)

    Parameters
    ----------
    k   : shape
    lam : scale
    pct : percentile, e.g. 10 for B10 life

    Returns
    -------
    float
    """
    p = pct / 100.0
    return lam * (-np.log(1 - p)) ** (1.0 / k)


# ---------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # True parameters
    TRUE_K = 2.5      # wear-out failure mode
    TRUE_LAM = 1000.0 # characteristic life: 1000 hours

    print(f"True distribution: Weibull(k={TRUE_K}, lam={TRUE_LAM})")
    print(f"True MTTF        : {mttf(TRUE_K, TRUE_LAM):.1f} hours")
    print(f"True B10 life    : {b_life(TRUE_K, TRUE_LAM, 10):.1f} hours")
    print(f"True B50 life    : {b_life(TRUE_K, TRUE_LAM, 50):.1f} hours")
    print()

    # Show what sparse data looks like
    samples = sparse_sample(TRUE_K, TRUE_LAM, seed=42)
    for n, data in samples.items():
        print(f"n={n:>4}  |  min={data.min():.0f}  mean={data.mean():.0f}  max={data.max():.0f}")

    print()

    # Censored example
    df = censored_sample(50, TRUE_K, TRUE_LAM, observation_time=800, seed=42)
    n_failed = df["failed"].sum()
    print(f"Censored sample (n=50, obs_time=800h):")
    print(f"  Failed  : {n_failed} ({100*n_failed/50:.0f}%)")
    print(f"  Censored: {50 - n_failed} ({100*(50-n_failed)/50:.0f}%)")
    print(df.head(10).to_string(index=False))
