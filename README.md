# reliability-toolkit

**Reliability analysis and automated assessment for sparse failure data.**

This toolkit addresses a core challenge in industrial reliability engineering:
*how do you characterize failure behavior — especially in the tail — when you only have a handful of observations?*

Built as a portfolio project targeting reliability/data science roles in semiconductor manufacturing and related fields.

---

## The Problem

When a component fails in the field, engineers need to estimate:
- **MTTF** — mean time to failure
- **B10 life** — time by which 10% of units will have failed
- **Tail probabilities** — P(T < t) for mission-critical thresholds

In mature product lines, thousands of failure records exist. But for new components, pilot deployments, or rare failure modes, you might have **5–15 data points**. Classical MLE breaks down here — confidence intervals blow up and tail estimates become unreliable.

This toolkit provides:
1. **Bayesian tail modeling** that incorporates engineering priors to regularize sparse estimates
2. **Automated distribution fitting and reporting** that scales across many components at once

---

## Project Structure

```
reliability-toolkit/
├── reliability/
│   ├── distributions.py     # Weibull, Lognormal, Pareto, Gumbel fitters (MLE + AIC/BIC)
│   ├── tail_analysis.py     # Bayesian Weibull model for sparse data (PyMC)
│   ├── simulate.py          # Synthetic data generation and censoring
│   └── report.py            # Automated report generator (batch-capable)
├── notebooks/
│   ├── 01_tail_modeling.ipynb     # Deep dive: sparse data + Bayesian updating
│   └── 02_reliability_tool.ipynb  # End-to-end automated reporting demo
├── tests/
│   ├── test_distributions.py
│   └── test_tail_analysis.py
├── data/                    # Sample and synthetic datasets
└── requirements.txt
```

---

## Key Concepts

### Why Weibull?

The Weibull distribution is the standard model for time-to-failure data because its shape parameter **k** directly encodes the failure mechanism:

| Shape (k) | Failure Mode | Physical Meaning |
|-----------|-------------|------------------|
| k < 1 | Infant mortality | Early defects, burn-in failures |
| k = 1 | Random failure | Memoryless, constant hazard rate |
| k > 1 | Wear-out | Fatigue, degradation over time |

A Weibull(k, λ) random variable T has survival function:

```
S(t) = P(T > t) = exp(-(t/λ)^k)
```

### The Sparse Data Problem

With n = 10 observations, MLE estimates of k and λ have enormous variance.
The 95% CI for the B10 life can span **an order of magnitude**.

This notebook quantifies exactly how bad that is, then shows how
Bayesian priors (derived from engineering specs or similar components)
dramatically tighten the estimates.

### Bayesian Updating

We place weakly informative priors on k and λ:

```python
with pm.Model():
    k = pm.LogNormal("k", mu=0, sigma=0.5)       # shape: centered near 1
    lam = pm.LogNormal("lam", mu=log(mu_prior), sigma=1)  # scale: from domain knowledge
    obs = pm.Weibull("obs", alpha=k, beta=lam, observed=data)
```

As more failures are observed, the posterior narrows toward the true value.
This is visualized explicitly in `01_tail_modeling.ipynb`.

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/reliability-toolkit.git
cd reliability-toolkit
pip install -r requirements.txt
jupyter notebook notebooks/01_tail_modeling.ipynb
```

### Run the automated report on your own data

```python
from reliability.report import ReliabilityReport

# CSV with a column called 'time_to_failure' (in hours)
report = ReliabilityReport.from_csv("data/my_component.csv")
report.fit()
report.summary()
# outputs: best distribution, MTTF, B10 life, tail probabilities, plots
```

### Batch assess a folder of components

```python
from reliability.report import batch_assess

results = batch_assess("data/field_components/")
results.to_csv("outputs/reliability_summary.csv")
```

---

## Notebooks

### `01_tail_modeling.ipynb` — Sparse Data & Bayesian Tail Analysis

- Simulate ground-truth Weibull data at large n
- Subsample to n = 5, 10, 15 and show MLE instability
- Fit Bayesian model and compare posterior predictive vs. ground truth
- Visualize how tail uncertainty shrinks as observations accumulate
- Prior sensitivity analysis: what happens when your prior is wrong?

### `02_reliability_tool.ipynb` — Automated Reporting Pipeline

- Load a dataset (real or synthetic)
- Auto-fit 5 candidate distributions, rank by AIC
- Generate probability plot, hazard rate, and survival function
- Output B10/B50 life estimates with credible intervals
- Batch mode: process a directory of component CSVs

---

## Roadmap

- [ ] Add right-censoring support (components still running at observation cutoff)
- [ ] Mixture models for competing failure modes
- [ ] AWS S3 integration for batch data ingestion
- [ ] Power BI / Streamlit dashboard for non-technical users
- [ ] Hierarchical Bayesian model (pool information across similar components)

---

## Background & Motivation

This project was built to explore the intersection of **actuarial methods** and **hardware reliability engineering** — specifically the challenge of estimating tail risk from limited field data, which is common in:

- Semiconductor capital equipment
- Aerospace components
- Medical devices
- Early-stage product deployments

The same mathematical framework used by insurance actuaries to price rare-event risk applies directly to predicting rare hardware failures.

---

## References

- Meeker & Escobar, *Statistical Methods for Reliability Data* (1998)
- Gelman et al., *Bayesian Data Analysis, 3rd ed.*
- ReliaSoft Weibull++ documentation
- NASA CMAPSS Turbofan Engine Degradation Dataset
- NIST/SEMATECH e-Handbook of Statistical Methods

---

## License

MIT
