import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Compute d1 and d2 for Black-Scholes with continuous dividend yield q."""
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Calculate Black-Scholes option price (vectorized, supports dividends).

    Parameters
    ----------
    S     : float or array  — Spot price
    K     : float or array  — Strike price
    T     : float or array  — Time to maturity (years)
    r     : float           — Risk-free rate (continuous)
    sigma : float or array  — Volatility
    option_type : 'call' or 'put'
    q     : float           — Continuous dividend yield (default 0)

    Returns
    -------
    price : float or array
    """
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))

    # Handle T = 0 (expiry) intrinsic value
    if np.ndim(T) == 0 and T <= 0:
        if option_type == 'call':
            return float(np.maximum(S - K, 0))
        return float(np.maximum(K - S, 0))

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


# ─── Greeks ───────────────────────────────────────────────────────────────────

def delta(S, K, T, r, sigma, option_type='call', q=0.0):
    """First derivative of price w.r.t. spot (Δ)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def gamma(S, K, T, r, sigma, q=0.0):
    """Second derivative of price w.r.t. spot (Γ) — same for calls and puts."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """First derivative of price w.r.t. volatility (ν) — per 1% move."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100


def theta(S, K, T, r, sigma, option_type='call', q=0.0):
    """Time decay — change in price per calendar day (Θ)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        return (common - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
    return (common + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365


def rho(S, K, T, r, sigma, option_type='call', q=0.0):
    """First derivative of price w.r.t. risk-free rate (ρ) — per 1% move."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


# ─── Second-order / cross Greeks ──────────────────────────────────────────────

def vanna(S, K, T, r, sigma, q=0.0):
    """dΔ/dσ = dν/dS  (sensitivity of delta to vol)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


def volga(S, K, T, r, sigma, q=0.0):
    """d²V/dσ²  (sensitivity of vega to vol, a.k.a. vomma) — per 1% move²."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return vega(S, K, T, r, sigma, q) * 100 * d1 * d2 / sigma


def charm(S, K, T, r, sigma, option_type='call', q=0.0):
    """dΔ/dt  (delta decay per calendar day)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == 'call':
        return (-np.exp(-q * T) * norm.pdf(d1) *
                (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) /
                (2 * T * sigma * np.sqrt(T))) / 365
    return (-np.exp(-q * T) * norm.pdf(d1) *
            (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) /
            (2 * T * sigma * np.sqrt(T))) / 365


def speed(S, K, T, r, sigma, q=0.0):
    """dΓ/dS  (rate of change of gamma with spot)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    g = gamma(S, K, T, r, sigma, q)
    return -g / S * (d1 / (sigma * np.sqrt(T)) + 1)


def all_greeks(S, K, T, r, sigma, option_type='call', q=0.0):
    """Return a dict of all first- and second-order Greeks."""
    return {
        'delta': delta(S, K, T, r, sigma, option_type, q),
        'gamma': gamma(S, K, T, r, sigma, q),
        'vega':  vega(S, K, T, r, sigma, q),
        'theta': theta(S, K, T, r, sigma, option_type, q),
        'rho':   rho(S, K, T, r, sigma, option_type, q),
        'vanna': vanna(S, K, T, r, sigma, q),
        'volga': volga(S, K, T, r, sigma, q),
        'charm': charm(S, K, T, r, sigma, option_type, q),
        'speed': speed(S, K, T, r, sigma, q),
    }