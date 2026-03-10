"""
censored.py
-----------
Survival analysis methods for right-censored time-to-failure data.

The Problem
-----------
Standard MLE assumes every observation is a failure. But real field data
has units still running at the observation cutoff — right-censored observations.
Ignoring them biases MTTF and B-life estimates downward.

This module handles censored data via:
    1. Kaplan-Meier estimator   — non-parametric survival curve
    2. Nelson-Aalen estimator   — cumulative hazard (more stable at small n)
    3. Weibull MLE with censoring — parametric fit using full likelihood:

        ℓ(k,λ) = Σ_failed log f(t_i) + Σ_censored log S(t_i)

        where f(t) = Weibull PDF, S(t) = Weibull survival function

Usage
-----
    from censored import CensoredSurvival

    df = pd.DataFrame({
        'time':   [120, 340, 500, 500, 500, 780, 900],
        'failed': [True, True, False, False, True, True, False]
    })

    cs = CensoredSurvival(df['time'], df['failed'], name='Bearing A')
    cs.fit()
    cs.summary()
    cs.plot()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma


# ---------------------------------------------------------------------
# Kaplan-Meier Estimator
# ---------------------------------------------------------------------

def kaplan_meier(times: np.ndarray, failed: np.ndarray) -> pd.DataFrame:
    """
    Kaplan-Meier non-parametric survival estimator.

    At each failure time t_i, updates survival estimate:
        S(t_i) = S(t_{i-1}) * (1 - d_i / n_i)

    where d_i = number of failures at t_i, n_i = number at risk just before t_i.

    Parameters
    ----------
    times  : observed times (failure or censoring)
    failed : boolean array, True if unit failed, False if censored

    Returns
    -------
    pd.DataFrame with columns: time, n_risk, n_failed, survival, std_err
    """
    times = np.asarray(times, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    n = len(times)

    # Sort by time
    order = np.argsort(times)
    times = times[order]
    failed = failed[order]

    # Only compute at failure times
    failure_times = np.unique(times[failed])

    rows = []
    S = 1.0
    var_sum = 0.0  # for Greenwood's formula

    for t in failure_times:
        at_risk_mask = times >= t
        n_risk = at_risk_mask.sum()
        n_fail = ((times == t) & failed).sum()

        if n_risk == 0:
            continue

        S_prev = S
        S = S * (1 - n_fail / n_risk)

        # Greenwood's formula for variance
        if n_risk > n_fail:
            var_sum += n_fail / (n_risk * (n_risk - n_fail))
        std_err = S * np.sqrt(var_sum)

        rows.append({
            "time": t,
            "n_risk": n_risk,
            "n_failed": n_fail,
            "survival": S,
            "std_err": std_err,
            "ci_low": max(0, S - 1.96 * std_err),
            "ci_high": min(1, S + 1.96 * std_err),
        })

    # Add t=0 anchor
    rows.insert(0, {"time": 0, "n_risk": n, "n_failed": 0,
                    "survival": 1.0, "std_err": 0.0,
                    "ci_low": 1.0, "ci_high": 1.0})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Nelson-Aalen Estimator
# ---------------------------------------------------------------------

def nelson_aalen(times: np.ndarray, failed: np.ndarray) -> pd.DataFrame:
    """
    Nelson-Aalen cumulative hazard estimator.

    At each failure time:
        H(t_i) = H(t_{i-1}) + d_i / n_i

    Survival estimate: S(t) = exp(-H(t))
    More stable than KM at small n and in the tail.

    Parameters
    ----------
    times  : observed times
    failed : boolean failure indicator

    Returns
    -------
    pd.DataFrame with columns: time, n_risk, n_failed, cum_hazard, survival
    """
    times = np.asarray(times, dtype=float)
    failed = np.asarray(failed, dtype=bool)

    order = np.argsort(times)
    times = times[order]
    failed = failed[order]

    failure_times = np.unique(times[failed])

    rows = []
    H = 0.0

    for t in failure_times:
        n_risk = (times >= t).sum()
        n_fail = ((times == t) & failed).sum()

        if n_risk == 0:
            continue

        H += n_fail / n_risk
        rows.append({
            "time": t,
            "n_risk": n_risk,
            "n_failed": n_fail,
            "cum_hazard": H,
            "survival": np.exp(-H),
        })

    rows.insert(0, {"time": 0, "n_risk": len(times), "n_failed": 0,
                    "cum_hazard": 0.0, "survival": 1.0})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Weibull MLE with Censoring
# ---------------------------------------------------------------------

def weibull_mle_censored(
    times: np.ndarray,
    failed: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """
    Fit Weibull distribution via MLE using the full censored likelihood.

    For complete (failed) observations, contribute log f(t):
        log f(t) = log k - k*log λ + (k-1)*log t - (t/λ)^k

    For censored observations, contribute log S(t):
        log S(t) = -(t/λ)^k

    Total log-likelihood:
        ℓ(k,λ) = Σ_failed log f(t_i) + Σ_censored log S(t_i)

    Parameters
    ----------
    times       : observed times (failures and censored)
    failed      : boolean failure indicator
    n_bootstrap : bootstrap resamples for CI
    seed        : random seed

    Returns
    -------
    dict with k, lam, aic, bic, k_ci, lam_ci, n_failed, n_censored
    """
    times = np.asarray(times, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    n = len(times)
    n_failed = failed.sum()
    n_censored = (~failed).sum()

    def neg_log_likelihood(params):
        log_k, log_lam = params
        k = np.exp(log_k)
        lam = np.exp(log_lam)

        # Failed observations: log f(t)
        ll_failed = 0.0
        if n_failed > 0:
            t_f = times[failed]
            ll_failed = np.sum(
                np.log(k) - k * np.log(lam) + (k - 1) * np.log(t_f) - (t_f / lam) ** k
            )

        # Censored observations: log S(t) = -(t/lam)^k
        ll_censored = 0.0
        if n_censored > 0:
            t_c = times[~failed]
            ll_censored = np.sum(-((t_c / lam) ** k))

        return -(ll_failed + ll_censored)

    # Initial guess: use sample mean for lam, k=1.5
    x0 = [np.log(1.5), np.log(np.mean(times))]
    result = minimize(neg_log_likelihood, x0, method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000})

    k = np.exp(result.x[0])
    lam = np.exp(result.x[1])
    ll = -result.fun

    # AIC / BIC (2 free parameters)
    p = 2
    aic = 2 * p - 2 * ll
    bic = p * np.log(n) - 2 * ll

    # Bootstrap CIs (resample pairs of (time, failed))
    rng = np.random.default_rng(seed)
    boot_k, boot_lam = [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        t_b, f_b = times[idx], failed[idx]
        if f_b.sum() < 2:
            continue
        try:
            x0_b = [np.log(1.5), np.log(np.mean(t_b))]
            res_b = minimize(
                lambda p: neg_log_likelihood.__wrapped__(p) if hasattr(neg_log_likelihood, '__wrapped__') else neg_log_likelihood(p),
                x0_b, method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000}
            )
            boot_k.append(np.exp(res_b.x[0]))
            boot_lam.append(np.exp(res_b.x[1]))
        except Exception:
            pass

    k_ci = np.percentile(boot_k, [2.5, 97.5]).tolist() if boot_k else [None, None]
    lam_ci = np.percentile(boot_lam, [2.5, 97.5]).tolist() if boot_lam else [None, None]

    return dict(
        k=k, lam=lam,
        aic=aic, bic=bic,
        ll=ll,
        k_ci=k_ci, lam_ci=lam_ci,
        n=n, n_failed=int(n_failed), n_censored=int(n_censored),
    )


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------

class CensoredSurvival:
    """
    Survival analysis pipeline for right-censored time-to-failure data.

    Combines:
        - Kaplan-Meier non-parametric estimator
        - Nelson-Aalen cumulative hazard estimator
        - Weibull MLE with full censored likelihood

    Usage
    -----
        cs = CensoredSurvival(times, failed, name="Bearing A")
        cs.fit()
        cs.summary()
        cs.plot()
    """

    def __init__(self, times, failed, name: str = "Component"):
        self.times = np.asarray(times, dtype=float)
        self.failed = np.asarray(failed, dtype=bool)
        self.name = name
        self.n = len(self.times)
        self._fitted = False

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       time_col: str = "time",
                       failed_col: str = "failed",
                       name: str = "Component"):
        return cls(df[time_col].values, df[failed_col].values, name=name)

    def fit(self, n_bootstrap: int = 500, seed: int = 42):
        """Run all three estimators."""
        self.km = kaplan_meier(self.times, self.failed)
        self.na = nelson_aalen(self.times, self.failed)
        self.weibull = weibull_mle_censored(
            self.times, self.failed,
            n_bootstrap=n_bootstrap, seed=seed
        )
        self._fitted = True

        # Derived metrics from Weibull fit
        k, lam = self.weibull["k"], self.weibull["lam"]
        self.mttf = lam * gamma(1 + 1.0 / k)
        self.b10 = lam * (-np.log(0.90)) ** (1.0 / k)
        self.b50 = lam * (-np.log(0.50)) ** (1.0 / k)

        return self

    def summary(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        w = self.weibull
        pct_censored = 100 * w["n_censored"] / w["n"]

        print("=" * 60)
        print(f"  CENSORED SURVIVAL ANALYSIS: {self.name}")
        print("=" * 60)
        print(f"  Total observations : {w['n']}")
        print(f"  Failures           : {w['n_failed']}  ({100 - pct_censored:.0f}%)")
        print(f"  Censored           : {w['n_censored']}  ({pct_censored:.0f}%)")
        print()
        print("  --- Weibull MLE (censored likelihood) ---")
        print(f"  Shape k   : {w['k']:.3f}  95% CI [{w['k_ci'][0]:.2f}, {w['k_ci'][1]:.2f}]")
        print(f"  Scale λ   : {w['lam']:.1f}  95% CI [{w['lam_ci'][0]:.0f}, {w['lam_ci'][1]:.0f}]")
        print(f"  AIC       : {w['aic']:.2f}")
        print()
        print("  --- Key Metrics ---")
        print(f"  MTTF      : {self.mttf:.1f} hrs")
        print(f"  B10 life  : {self.b10:.1f} hrs")
        print(f"  B50 life  : {self.b50:.1f} hrs")

        if w["k"] < 1:
            mode = "infant mortality"
        elif w["k"] < 1.5:
            mode = "near-random failure"
        else:
            mode = "wear-out"
        print(f"  Failure mode: {mode}  (k={w['k']:.2f})")
        print("=" * 60)

    def plot(self, true_k=None, true_lam=None, save_path=None):
        """
        4-panel plot:
            1. KM survival curve with 95% CI band
            2. Nelson-Aalen cumulative hazard
            3. KM vs Weibull parametric fit comparison
            4. Censoring overview (timeline)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        k, lam = self.weibull["k"], self.weibull["lam"]
        t_max = self.times.max() * 1.2
        t = np.linspace(0.01, t_max, 300)
        weibull_survival = np.exp(-((t / lam) ** k))
        weibull_hazard = (k / lam) * (t / lam) ** (k - 1)

        fig = plt.figure(figsize=(13, 10))
        fig.suptitle(
            f"Censored Survival Analysis: {self.name}  "
            f"(n={self.n}, {self.weibull['n_failed']} failures, "
            f"{self.weibull['n_censored']} censored)",
            fontsize=12, fontweight="bold"
        )
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # --- 1. KM survival curve ---
        ax1 = fig.add_subplot(gs[0, 0])
        km = self.km
        ax1.step(km["time"], km["survival"], where="post",
                 color="steelblue", lw=2, label="Kaplan-Meier")
        ax1.fill_between(km["time"], km["ci_low"], km["ci_high"],
                         step="post", alpha=0.2, color="steelblue", label="95% CI")
        if true_k and true_lam:
            ax1.plot(t, np.exp(-((t / true_lam) ** true_k)),
                     "r--", lw=1.5, label=f"True (k={true_k})")
        # Censored tick marks
        censored_times = self.times[~self.failed]
        ax1.plot(censored_times, np.full(len(censored_times), 0.02),
                 "+", ms=8, color="gray", alpha=0.6, label="Censored")
        ax1.set_xlabel("Time (hrs)")
        ax1.set_ylabel("S(t)")
        ax1.set_title("Kaplan-Meier Survival Curve")
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # --- 2. Nelson-Aalen cumulative hazard ---
        ax2 = fig.add_subplot(gs[0, 1])
        na = self.na
        ax2.step(na["time"], na["cum_hazard"], where="post",
                 color="crimson", lw=2, label="Nelson-Aalen H(t)")
        ax2.plot(t, (t / lam) ** k, "--", color="darkorange",
                 lw=1.5, label=f"Weibull H(t) fit")
        ax2.set_xlabel("Time (hrs)")
        ax2.set_ylabel("H(t)")
        ax2.set_title("Cumulative Hazard (Nelson-Aalen)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # --- 3. KM vs Weibull comparison ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.step(km["time"], km["survival"], where="post",
                 color="steelblue", lw=2, label="Kaplan-Meier")
        ax3.plot(t, weibull_survival, color="darkorange", lw=2,
                 label=f"Weibull fit (k={k:.2f}, λ={lam:.0f})")
        if true_k and true_lam:
            ax3.plot(t, np.exp(-((t / true_lam) ** true_k)),
                     "r--", lw=1.5, label="True")
        ax3.axvline(self.b10, color="gray", linestyle=":",
                    alpha=0.8, label=f"B10={self.b10:.0f}")
        ax3.set_xlabel("Time (hrs)")
        ax3.set_ylabel("S(t)")
        ax3.set_title("KM vs Weibull Parametric Fit")
        ax3.set_ylim(0, 1.05)
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # --- 4. Censoring overview ---
        ax4 = fig.add_subplot(gs[1, 1])
        order = np.argsort(self.times)
        sorted_times = self.times[order]
        sorted_failed = self.failed[order]

        colors = ["steelblue" if f else "lightgray" for f in sorted_failed]
        markers = ["x" if f else "|" for f in sorted_failed]

        for i, (t_i, f_i, c) in enumerate(zip(sorted_times, sorted_failed, colors)):
            ax4.plot([0, t_i], [i, i], color="lightgray", lw=0.5, alpha=0.5)
            ax4.plot(t_i, i, "x" if f_i else "|",
                     color="steelblue" if f_i else "gray",
                     ms=6, mew=1.5)

        ax4.set_xlabel("Time (hrs)")
        ax4.set_ylabel("Unit index (sorted)")
        ax4.set_title("Observation Timeline\n(✕ = failure, | = censored)")
        ax4.grid(alpha=0.2, axis="x")

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="x", color="steelblue", lw=0,
                   markersize=7, label=f"Failed ({self.weibull['n_failed']})"),
            Line2D([0], [0], marker="|", color="gray", lw=0,
                   markersize=7, label=f"Censored ({self.weibull['n_censored']})"),
        ]
        ax4.legend(handles=legend_elements, fontsize=9)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()
        return fig


# ---------------------------------------------------------------------
# Bias demonstration: censored vs naive MLE
# ---------------------------------------------------------------------

def censoring_bias_demo(
    true_k: float = 2.5,
    true_lam: float = 1000.0,
    n: int = 50,
    observation_time: float = 800.0,
    seed: int = 42,
):
    """
    Show the bias introduced by ignoring censored observations.

    Compares three estimators on the same censored dataset:
        1. Naive MLE — drops censored observations entirely (BIASED)
        2. Censored MLE — uses full likelihood (UNBIASED)
        3. True values — ground truth for reference
    """
    import sys
    sys.path.append(".")
    from simulate import censored_sample
    from distributions import fit_weibull

    df = censored_sample(n, true_k, true_lam, observation_time, seed=seed)
    times = df["time"].values
    failed = df["failed"].values

    n_failed = failed.sum()
    n_censored = (~failed).sum()

    # Naive: fit only on failures, ignore censored
    naive = fit_weibull(times[failed], seed=seed)

    # Correct: use censored likelihood
    correct = weibull_mle_censored(times, failed, seed=seed)

    # True B10
    true_b10 = true_lam * (-np.log(0.90)) ** (1.0 / true_k)
    naive_b10 = naive["lam"] * (-np.log(0.90)) ** (1.0 / naive["k"])
    correct_b10 = correct["lam"] * (-np.log(0.90)) ** (1.0 / correct["k"])

    print("=" * 60)
    print(f"  CENSORING BIAS DEMONSTRATION")
    print(f"  n={n}, obs_time={observation_time}, "
          f"failed={n_failed}, censored={n_censored}")
    print("=" * 60)
    print(f"  {'Method':<20} {'k':>8} {'λ':>10} {'B10':>10}")
    print(f"  {'-'*48}")
    print(f"  {'True values':<20} {true_k:>8.3f} {true_lam:>10.1f} {true_b10:>10.1f}")
    print(f"  {'Naive MLE (biased)':<20} {naive['k']:>8.3f} {naive['lam']:>10.1f} {naive_b10:>10.1f}")
    print(f"  {'Censored MLE':<20} {correct['k']:>8.3f} {correct['lam']:>10.1f} {correct_b10:>10.1f}")
    print()
    print(f"  Naive B10 error  : {abs(naive_b10 - true_b10):.1f} hrs "
          f"({100*abs(naive_b10 - true_b10)/true_b10:.1f}% off)")
    print(f"  Correct B10 error: {abs(correct_b10 - true_b10):.1f} hrs "
          f"({100*abs(correct_b10 - true_b10)/true_b10:.1f}% off)")
    print("=" * 60)


# ---------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from simulate import censored_sample

    TRUE_K, TRUE_LAM = 2.5, 1000.0
    OBS_TIME = 800.0

    print("Generating censored field data...")
    df = censored_sample(60, TRUE_K, TRUE_LAM, OBS_TIME, seed=42)
    print(f"  {df['failed'].sum()} failures, {(~df['failed']).sum()} censored\n")

    cs = CensoredSurvival.from_dataframe(df, name="Bearing A")
    cs.fit()
    cs.summary()
    cs.plot(true_k=TRUE_K, true_lam=TRUE_LAM, save_path="censored_survival.png")

    print("\n--- Bias demonstration ---\n")
    censoring_bias_demo(TRUE_K, TRUE_LAM, n=60, observation_time=OBS_TIME)
