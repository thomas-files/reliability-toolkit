"""
report.py
---------
Automated reliability assessment report generator.

Takes raw time-to-failure data as input and outputs:
    - Best-fit distribution (by AIC)
    - MTTF with confidence interval
    - B10 / B50 life estimates
    - Tail probability estimates
    - Summary plots
    - Plain-English interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.special import gamma

from distributions import (
    fit_weibull,
    fit_lognormal,
    fit_exponential,
    compare_distributions,
    weibull_survival,
    weibull_b_life,
    weibull_mttf,
)


class ReliabilityReport:
    """
    Automated reliability assessment for a single component dataset.

    Usage
    -----
    report = ReliabilityReport(data, component_name="Bearing A")
    report.fit()
    report.summary()
    report.plot()
    """

    def __init__(self, data: np.ndarray, component_name: str = "Component"):
        """
        Parameters
        ----------
        data           : 1D array of positive failure times
        component_name : label used in plots and summaries
        """
        self.data = np.asarray(data, dtype=float)
        self.name = component_name
        self.n = len(self.data)
        self._fitted = False

    @classmethod
    def from_csv(cls, path: str, time_col: str = "time_to_failure", **kwargs):
        """
        Load failure data from a CSV file.

        CSV must have a column named `time_col` (default: 'time_to_failure').

        Parameters
        ----------
        path     : path to CSV file
        time_col : column name containing failure times
        """
        df = pd.read_csv(path)
        if time_col not in df.columns:
            raise ValueError(
                f"Column '{time_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        name = Path(path).stem
        return cls(df[time_col].dropna().values, component_name=name, **kwargs)

    def fit(self, n_bootstrap: int = 500, seed: int = 42):
        """
        Fit all candidate distributions and select best by AIC.

        Populates:
            self.comparison   : DataFrame of all fits ranked by AIC
            self.best_fit     : dict with best distribution params
            self.best_name    : name of best distribution
            self.metrics      : dict with MTTF, B10, B50, tail probs
        """
        self.comparison = compare_distributions(self.data, seed=seed)
        best_row = self.comparison.iloc[0]
        self.best_name = best_row["distribution"]
        self.best_params = best_row["params"]

        # Refit best distribution to get full results
        if self.best_name == "Weibull":
            self._best = fit_weibull(self.data, n_bootstrap=n_bootstrap, seed=seed)
        elif self.best_name == "Lognormal":
            self._best = fit_lognormal(self.data, n_bootstrap=n_bootstrap, seed=seed)
        else:
            self._best = fit_exponential(self.data)

        # Compute metrics (Weibull-based for now)
        if self.best_name in ("Weibull", "Exponential"):
            k = self._best.get("k", 1.0)
            lam = self._best["lam"]
            self.metrics = {
                "mttf": weibull_mttf(k, lam),
                "b10": weibull_b_life(k, lam, 10),
                "b50": weibull_b_life(k, lam, 50),
                "p_fail_before_mttf": 1 - float(
                    np.exp(-((weibull_mttf(k, lam) / lam) ** k))
                ),
            }
        else:
            # Lognormal metrics
            mu = self._best["mu"]
            sigma = self._best["sigma"]
            self.metrics = {
                "mttf": float(np.exp(mu + 0.5 * sigma**2)),
                "b10": float(np.exp(mu + sigma * stats.norm.ppf(0.10))),
                "b50": float(np.exp(mu + sigma * stats.norm.ppf(0.50))),
                "p_fail_before_mttf": 0.5,
            }

        self._fitted = True
        return self

    def summary(self):
        """Print a plain-English reliability summary."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .summary()")

        print("=" * 60)
        print(f"  RELIABILITY REPORT: {self.name}")
        print("=" * 60)
        print(f"  Sample size       : {self.n} failures")
        print(f"  Best-fit model    : {self.best_name}")
        print(f"  AIC               : {self.comparison.iloc[0]['aic']:.2f}")
        print(f"  KS p-value        : {self.comparison.iloc[0]['ks_pvalue']:.4f}")
        print()
        print("  --- Key Reliability Metrics ---")
        print(f"  MTTF (mean life)  : {self.metrics['mttf']:>10.1f} hrs")
        print(f"  B10 life          : {self.metrics['b10']:>10.1f} hrs")
        print(f"  B50 life          : {self.metrics['b50']:>10.1f} hrs")
        print()

        if self.best_name == "Weibull":
            k = self._best["k"]
            lam = self._best["lam"]
            k_ci = self._best["k_ci"]
            lam_ci = self._best["lam_ci"]
            print(f"  --- Weibull Parameters ---")
            print(f"  Shape k           : {k:.3f}  95% CI [{k_ci[0]:.2f}, {k_ci[1]:.2f}]")
            print(f"  Scale λ           : {lam:.1f}  95% CI [{lam_ci[0]:.0f}, {lam_ci[1]:.0f}]")
            print()
            if k < 1:
                mode = "infant mortality (early defects, burn-in failures)"
            elif k < 1.5:
                mode = "near-random failure (weak wear-out)"
            else:
                mode = "wear-out failure (fatigue/degradation over time)"
            print(f"  Failure mode      : {mode}")

        print()
        print("  --- Distribution Comparison ---")
        print(
            self.comparison[["rank", "distribution", "aic", "bic", "ks_pvalue"]]
            .to_string(index=False)
        )
        print("=" * 60)

    def plot(self, save_path: str = None):
        """
        Generate 4-panel reliability plot:
            1. Probability plot (linearizes the CDF for visual fit check)
            2. Survival function S(t) = P(T > t)
            3. Hazard rate h(t) = f(t) / S(t)
            4. PDF f(t)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .plot()")

        if self.best_name == "Weibull":
            k = self._best["k"]
            lam = self._best["lam"]
        elif self.best_name == "Exponential":
            k = 1.0
            lam = self._best["lam"]
        else:
            # Approximate lognormal with Weibull for plotting
            k = 2.0
            lam = self.metrics["mttf"]

        t = np.linspace(0.01, 3 * lam, 300)

        survival = np.exp(-((t / lam) ** k))
        hazard = (k / lam) * (t / lam) ** (k - 1)
        pdf = hazard * survival

        fig = plt.figure(figsize=(13, 10))
        fig.suptitle(f"Reliability Analysis: {self.name}  (n={self.n})",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

        # --- 1. Probability plot ---
        ax1 = fig.add_subplot(gs[0, 0])
        sorted_data = np.sort(self.data)
        empirical_cdf = (np.arange(1, self.n + 1) - 0.3) / (self.n + 0.4)
        if self.best_name in ("Weibull", "Exponential"):
            # Weibull probability plot: ln(t) vs ln(-ln(1-F))
            x = np.log(sorted_data)
            y = np.log(-np.log(1 - empirical_cdf))
            x_fit = np.log(t)
            y_fit = np.log(-np.log(survival))
            ax1.scatter(x, y, color="steelblue", s=30, zorder=5, label="Observed")
            ax1.plot(x_fit, y_fit, color="crimson", lw=2, label="Weibull fit")
            ax1.set_xlabel("ln(t)")
            ax1.set_ylabel("ln(-ln(1-F))")
            ax1.set_title("Weibull Probability Plot")
        else:
            ax1.scatter(sorted_data, empirical_cdf,
                        color="steelblue", s=30, label="Empirical CDF")
            ax1.set_xlabel("t")
            ax1.set_ylabel("F(t)")
            ax1.set_title("Probability Plot")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # --- 2. Survival function ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, survival, color="steelblue", lw=2.5)
        ax2.axvline(self.metrics["mttf"], color="gray", linestyle="--",
                    alpha=0.7, label=f"MTTF={self.metrics['mttf']:.0f}")
        ax2.axvline(self.metrics["b10"], color="orange", linestyle=":",
                    alpha=0.9, label=f"B10={self.metrics['b10']:.0f}")
        ax2.plot(self.data, np.zeros(self.n), "|", color="black",
                 ms=8, alpha=0.5, label="Failures")
        ax2.set_xlabel("Time (hrs)")
        ax2.set_ylabel("S(t) = P(T > t)")
        ax2.set_title("Survival Function")
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # --- 3. Hazard rate ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(t, hazard, color="crimson", lw=2.5)
        ax3.set_xlabel("Time (hrs)")
        ax3.set_ylabel("h(t)")
        ax3.set_title("Hazard Rate")
        if k < 1:
            ax3.annotate("Decreasing: infant mortality",
                         xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8)
        elif k == 1:
            ax3.annotate("Constant: random failure",
                         xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8)
        else:
            ax3.annotate("Increasing: wear-out",
                         xy=(0.05, 0.85), xycoords="axes fraction", fontsize=8)
        ax3.grid(alpha=0.3)

        # --- 4. PDF ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(t, pdf, color="seagreen", lw=2.5)
        ax4.hist(self.data, bins="auto", density=True,
                 alpha=0.3, color="seagreen", label="Observed")
        ax4.set_xlabel("Time (hrs)")
        ax4.set_ylabel("f(t)")
        ax4.set_title("Failure Density (PDF)")
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")

        plt.show()
        return fig


# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------

def batch_assess(data_dir: str, time_col: str = "time_to_failure") -> pd.DataFrame:
    """
    Run ReliabilityReport on every CSV in a directory.

    Parameters
    ----------
    data_dir : path to folder containing CSV files
    time_col : column name for failure times in each CSV

    Returns
    -------
    pd.DataFrame with one row per component:
        component, n, best_dist, aic, mttf, b10, b50
    """
    rows = []
    for csv_path in sorted(Path(data_dir).glob("*.csv")):
        try:
            report = ReliabilityReport.from_csv(str(csv_path), time_col=time_col)
            report.fit()
            rows.append({
                "component": csv_path.stem,
                "n": report.n,
                "best_dist": report.best_name,
                "aic": round(report.comparison.iloc[0]["aic"], 2),
                "mttf": round(report.metrics["mttf"], 1),
                "b10_life": round(report.metrics["b10"], 1),
                "b50_life": round(report.metrics["b50"], 1),
            })
        except Exception as e:
            print(f"Warning: failed on {csv_path.name}: {e}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from simulate import weibull_sample, censored_sample

    TRUE_K = 2.5
    TRUE_LAM = 1000.0

    # Single component report
    print("Running single component report (n=50)...\n")
    data = weibull_sample(50, TRUE_K, TRUE_LAM, seed=42)
    report = ReliabilityReport(data, component_name="Bearing A")
    report.fit()
    report.summary()
    report.plot(save_path="reliability_report.png")

    # Batch demo — generate 3 synthetic components
    print("\nRunning batch assessment...\n")
    import os, tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        components = [
            ("bearing_a", 2.5, 1000),
            ("motor_b",   1.2,  500),
            ("sensor_c",  3.0, 2000),
        ]
        for name, k, lam in components:
            d = weibull_sample(30, k, lam, seed=42)
            pd.DataFrame({"time_to_failure": d}).to_csv(
                f"{tmpdir}/{name}.csv", index=False
            )

        results = batch_assess(tmpdir)
        print(results.to_string(index=False))
