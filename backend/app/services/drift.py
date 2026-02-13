import numpy as np
from scipy import stats

def psi(expected, actual, bins=10):
    breakpoints = np.linspace(0, 1, bins + 1)
    e_hist, _ = np.histogram(expected, bins=breakpoints)
    a_hist, _ = np.histogram(actual, bins=breakpoints)

    e_perc = e_hist / max(len(expected), 1)
    a_perc = a_hist / max(len(actual), 1)

    return float(np.sum((a_perc - e_perc) * np.log((a_perc + 1e-10) / (e_perc + 1e-10))))

def ks(expected, actual):
    return float(stats.ks_2samp(expected, actual).statistic)

def severity_from_psi(v):
    return "high" if v > 0.5 else "medium" if v > 0.25 else "low"

def score_drift(reference_scores, current_scores):
    vpsi = psi(reference_scores, current_scores)
    vks = ks(reference_scores, current_scores)
    drift = (vpsi > 0.25) or (vks > 0.1)
    return {"psi": vpsi, "ks": vks, "drift_detected": drift, "severity": severity_from_psi(vpsi)}