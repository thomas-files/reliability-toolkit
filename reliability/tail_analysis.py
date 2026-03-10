"""
tail_analysis.py
----------------
Bayesian and frequentist methods for tail characterization
under sparse data conditions (n = 5–30 observations).

Key functions:
    - sparse_weibull_bayes()   : PyMC model with informative priors
    - tail_quantile_ci()       : credible intervals for P(T > t)
    - prior_sensitivity()      : how much does prior choice matter?
    - plot_tail_uncertainty()  : visualize posterior predictive tail
"""

# TODO: implement Bayesian Weibull model using PyMC
# TODO: implement frequentist bootstrap as comparison baseline
