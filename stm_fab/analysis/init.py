# ti_fitter.py
import numpy as np
from scipy.optimize import curve_fit

def model_exp(I, a, b, c):   # T = a + b * exp(c * I)
    return a + b * np.exp(c * I)

def model_pow(I, a, b, c):   # T = a + b * I^c
    return a + b * np.power(I, c)

def model_log(I, a, b):      # T = a + b * ln(I)
    return a + b * np.log(I)

def aic(n, rss, k):
    return n*np.log(rss/n) + 2*k

def fit_T_of_I(I, T):
    # Filter finite and positive for log/power stability
    mask = np.isfinite(I) & np.isfinite(T) & (I>0)
    I = I[mask]; T = T[mask]
    n = len(I)
    results = []

    # Exponential
    try:
        p0 = [np.median(T), (np.max(T)-np.min(T))/2, 2.0]
        popt, pcov = curve_fit(model_exp, I, T, p0=p0, maxfev=20000)
        residuals = T - model_exp(I, *popt)
        rss = np.sum(residuals**2)
        results.append(('exp', popt, pcov, rss, aic(n, rss, 3)))
    except:
        pass

    # Power
    try:
        p0 = [np.min(T), (np.max(T)-np.min(T)), 1.0]
        popt, pcov = curve_fit(model_pow, I, T, p0=p0, bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 5.0]), maxfev=20000)
        residuals = T - model_pow(I, *popt)
        rss = np.sum(residuals**2)
        results.append(('pow', popt, pcov, rss, aic(n, rss, 3)))
    except:
        pass

    # Log
    try:
        p0 = [np.median(T), (np.max(T)-np.min(T))/np.log(np.max(I)/np.min(I))]
        popt, pcov = curve_fit(model_log, I, T, p0=p0, maxfev=20000)
        residuals = T - model_log(I, *popt)
        rss = np.sum(residuals**2)
        results.append(('log', popt, pcov, rss, aic(n, rss, 2)))
    except:
        pass

    if not results:
        raise ValueError("No model could be fitted")

    # Pick by AIC (lower is better)
    best = min(results, key=lambda r: r[4])
    kind, popt, pcov, rss, aic_val = best
    return {
        'model': kind,
        'params': popt,
        'cov': pcov,
        'rss': rss,
        'aic': aic_val
    }
