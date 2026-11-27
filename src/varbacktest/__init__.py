"""One-day VaR and ES estimation and backtesting for a buy-and-hold equity portfolio.

Conventions used throughout the package:
    - simple returns; loss L_t = -r_t;
    - VaR and ES are positive numbers, expressed as a fraction of portfolio value;
    - the forecast for day t uses only data up to day t-1.
"""
