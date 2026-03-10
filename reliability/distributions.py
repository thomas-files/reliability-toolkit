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
    - 95% confidence intervals (via Fisher information or bootstrap)
    - AIC / BIC for model comparison
    - Goodness-of-fit p-value (KS test)
"""

# TODO: implement WeibullFitter, LognormalFitter, ParetofFitter
# TODO: implement compare_distributions() -> ranked DataFrame by AIC
