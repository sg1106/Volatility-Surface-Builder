"""
surface_model.py — Volatility surface fitting and arbitrage diagnostics.

Provides:
  • SVI (Stochastic Volatility Inspired) parametrization per expiry slice
  • Cubic-spline interpolation across the full surface
  • Calendar- and butterfly-spread arbitrage detection
  • Moneyness normalisation helpers
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import SmoothBivariateSpline, RectBivariateSpline


# ─── Moneyness ────────────────────────────────────────────────────────────────

def log_moneyness(S, K, T, r, q=0.0):
    """Log-forward moneyness:  k = log(K / F)  where F = S·exp((r-q)T)."""
    F = S * np.exp((r - q) * T)
    return np.log(K / F)


def moneyness_label(k, n=5):
    """Return a human-readable moneyness label for a log-moneyness value k."""
    if abs(k) < 0.01:
        return 'ATM'
    side = 'OTM' if k > 0 else 'ITM'
    pct = f'{abs(k)*100:.1f}%'
    return f'{side} {pct}'


# ─── SVI raw parametrization ──────────────────────────────────────────────────
# w(k) = a + b·( ρ·(k-m) + √((k-m)² + σ²) )
# IV(k) = √(w(k)/T)

def svi_raw(k, a, b, rho, m, sigma):
    """SVI total variance w(k)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi_slice(k_arr, iv_arr, T, w0=None):
    """
    Fit SVI parameters to a single expiry slice.

    Parameters
    ----------
    k_arr : array  — Log-forward moneyness values
    iv_arr: array  — Implied volatility values
    T     : float  — Time to maturity
    w0    : dict   — Initial guess {a,b,rho,m,sigma}

    Returns
    -------
    params: dict or None
    """
    w_arr = iv_arr ** 2 * T  # total variance

    def loss(p):
        a, b, rho, m, sigma = p
        w_fit = svi_raw(k_arr, a, b, rho, m, sigma)
        return np.sum((w_fit - w_arr) ** 2)

    # Sensible default seed
    a0 = np.mean(w_arr) * 0.8
    b0 = 0.1
    rho0 = -0.3
    m0 = 0.0
    sigma0 = 0.2

    x0 = [a0, b0, rho0, m0, sigma0] if w0 is None else [
        w0['a'], w0['b'], w0['rho'], w0['m'], w0['sigma']
    ]

    bounds = [
        (1e-6, None),   # a >= 0
        (1e-6, None),   # b > 0
        (-0.999, 0.999),# |rho| < 1
        (-1.0, 1.0),    # m
        (1e-6, None),   # sigma > 0
    ]

    res = minimize(loss, x0, bounds=bounds, method='L-BFGS-B',
                   options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 500})

    if not res.success and res.fun > 1e-3:
        return None

    a, b, rho, m, sigma = res.x
    return dict(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T, loss=res.fun)


def svi_iv(k, params):
    """Evaluate fitted SVI IV at log-moneyness k."""
    w = svi_raw(k, params['a'], params['b'], params['rho'],
                params['m'], params['sigma'])
    w = np.maximum(w, 0)
    return np.sqrt(w / params['T'])


# ─── Spline surface ───────────────────────────────────────────────────────────

class VolatilitySurface:
    """
    Smooth volatility surface fitted via bivariate spline.

    Usage
    -----
        surf = VolatilitySurface()
        surf.fit(strikes, maturities, ivs, spot, r)
        iv = surf.predict(K=150, T=0.5)
    """

    def __init__(self):
        self._spline = None
        self._spot = None
        self._r = None
        self._q = None
        self.svi_slices = {}   # T → SVI params

    def fit(self, strikes, maturities, ivs, spot, r, q=0.0, kx=3, ky=3):
        """
        Fit a smooth spline to (strike, maturity, IV) triples.

        Parameters
        ----------
        strikes    : array-like
        maturities : array-like
        ivs        : array-like
        spot       : float
        r          : float
        q          : float
        kx, ky     : int  — Spline degrees (3 = cubic)
        """
        self._spot, self._r, self._q = spot, r, q

        strikes = np.asarray(strikes, float)
        maturities = np.asarray(maturities, float)
        ivs = np.asarray(ivs, float)

        mask = np.isfinite(ivs) & (ivs > 0) & (ivs < 5)
        strikes, maturities, ivs = strikes[mask], maturities[mask], ivs[mask]

        try:
            self._spline = SmoothBivariateSpline(
                strikes, maturities, ivs, kx=kx, ky=ky, s=len(ivs) * 0.01
            )
        except Exception as e:
            print(f'[VolatilitySurface] Spline fit failed: {e}')
            self._spline = None

        # Fit SVI per slice
        for T in np.unique(maturities):
            mask_T = maturities == T
            k_arr = log_moneyness(spot, strikes[mask_T], T, r, q)
            iv_arr = ivs[mask_T]
            if len(k_arr) >= 5:
                params = fit_svi_slice(k_arr, iv_arr, T)
                if params:
                    self.svi_slices[T] = params

    def predict(self, K, T):
        """Predict IV for given strike K and maturity T."""
        if self._spline is None:
            return np.nan
        val = self._spline(K, T)
        return float(np.clip(val, 0.01, 5.0))

    def grid(self, n_strikes=60, n_maturities=30, strike_range=0.4):
        """Return (K_grid, T_grid, IV_grid) for surface plotting."""
        if self._spline is None or self._spot is None:
            return None
        spot = self._spot
        K_vals = np.linspace(spot * (1 - strike_range), spot * (1 + strike_range), n_strikes)
        T_min, T_max = min(self.svi_slices) if self.svi_slices else 0.05, 2.0
        T_vals = np.linspace(T_min, T_max, n_maturities)
        K_grid, T_grid = np.meshgrid(K_vals, T_vals)
        IV_grid = np.array([[self.predict(k, t) for k in K_vals] for t in T_vals])
        return K_grid, T_grid, IV_grid


# ─── Arbitrage checks ─────────────────────────────────────────────────────────

def check_put_call_parity(call_price, put_price, S, K, T, r, q=0.0, tol=0.02):
    """
    Check put-call parity:  C - P ≈ S·e^{-qT} - K·e^{-rT}

    Returns a dict with theoretical difference, actual difference, and flag.
    """
    theoretical = S * np.exp(-q * T) - K * np.exp(-r * T)
    actual = call_price - put_price
    error = actual - theoretical
    return {
        'theoretical': theoretical,
        'actual': actual,
        'error': error,
        'violation': abs(error) > tol * S,
    }


def detect_butterfly_arb(iv_slice):
    """
    Detect butterfly arbitrage in a single expiry slice.

    A butterfly arbitrage exists when the smile is locally concave in total
    variance space (Gatheral's condition): d²w/dk² < 0.

    Parameters
    ----------
    iv_slice : list[dict]  — Each dict has 'strike' and 'iv' keys, sorted by strike.

    Returns
    -------
    violations : list[dict]  — Strikes where potential arbitrage is detected.
    """
    if len(iv_slice) < 3:
        return []

    iv_slice = sorted(iv_slice, key=lambda x: x['strike'])
    violations = []
    T = iv_slice[0].get('T', 1.0)

    for i in range(1, len(iv_slice) - 1):
        k_prev = iv_slice[i - 1]['strike']
        k_curr = iv_slice[i]['strike']
        k_next = iv_slice[i + 1]['strike']
        w_prev = iv_slice[i - 1]['iv'] ** 2 * T
        w_curr = iv_slice[i]['iv'] ** 2 * T
        w_next = iv_slice[i + 1]['iv'] ** 2 * T

        # Finite-difference second derivative of w
        d2w = (w_next - 2 * w_curr + w_prev) / ((k_next - k_prev) / 2) ** 2
        if d2w < -1e-4:
            violations.append({
                'strike': k_curr,
                'iv': iv_slice[i]['iv'],
                'd2w': d2w,
            })

    return violations


def detect_calendar_arb(slices_by_T):
    """
    Detect calendar arbitrage: total variance must be non-decreasing in T
    at the same strike.

    Parameters
    ----------
    slices_by_T : dict {T: list[dict]}  — Each inner list has 'strike' and 'iv'.

    Returns
    -------
    violations : list[dict]
    """
    sorted_Ts = sorted(slices_by_T.keys())
    violations = []

    for i in range(1, len(sorted_Ts)):
        T1, T2 = sorted_Ts[i - 1], sorted_Ts[i]
        slice1 = {d['strike']: d['iv'] for d in slices_by_T[T1]}
        slice2 = {d['strike']: d['iv'] for d in slices_by_T[T2]}
        common = set(slice1) & set(slice2)
        for K in common:
            w1 = slice1[K] ** 2 * T1
            w2 = slice2[K] ** 2 * T2
            if w2 < w1 - 1e-6:
                violations.append({
                    'strike': K,
                    'T1': T1, 'T2': T2,
                    'w1': w1, 'w2': w2,
                })

    return violations