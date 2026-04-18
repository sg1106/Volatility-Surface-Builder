import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from black_scholes import black_scholes_price, vega as bs_vega


# ─── Analytical vega for Newton step ──────────────────────────────────────────

def _intrinsic(S, K, T, r, option_type):
    """Minimum possible option price (intrinsic + tiny time value floor)."""
    fwd = S * np.exp(r * T)
    if option_type == 'call':
        return max(np.exp(-r * T) * max(fwd - K, 0), 0)
    return max(np.exp(-r * T) * max(K - fwd, 0), 0)


def _brenner_subrahmanyam_seed(price, S, K, T, r):
    """Quick ATM approximation used as Newton seed: σ ≈ √(2π/T) · (P/S)."""
    return np.sqrt(2 * np.pi / T) * price / S


# ─── Newton-Raphson with Brent's fallback ────────────────────────────────────

def implied_volatility(
    price, S, K, T, r,
    option_type='call',
    q=0.0,
    tol=1e-7,
    max_iter_nr=50,
    max_iter_brent=200,
):
    """
    Implied volatility via Newton-Raphson (fast convergence near ATM)
    with automatic fallback to Brent's method for robustness.

    Parameters
    ----------
    price       : float  — Market option price
    S           : float  — Spot price
    K           : float  — Strike price
    T           : float  — Time to maturity (years)
    r           : float  — Risk-free rate
    option_type : 'call' or 'put'
    q           : float  — Continuous dividend yield
    tol         : float  — Convergence tolerance
    max_iter_nr : int    — Max Newton-Raphson iterations
    max_iter_brent: int  — Max Brent iterations

    Returns
    -------
    iv : float or None
    """
    if T <= 0 or price <= 0:
        return None

    intrinsic = _intrinsic(S, K, T, r, option_type)
    upper_bound = S * np.exp(-q * T)  # max call price
    if price < intrinsic or price >= upper_bound * 1.05:
        return None

    # ── Newton-Raphson ──────────────────────────────────────────────────────
    sigma = _brenner_subrahmanyam_seed(price, S, K, T, r)
    sigma = np.clip(sigma, 1e-4, 5.0)

    for _ in range(max_iter_nr):
        try:
            model_price = black_scholes_price(S, K, T, r, sigma, option_type, q)
            v = bs_vega(S, K, T, r, sigma, q) * 100  # undo /100 in vega()
            if v < 1e-10:
                break
            diff = model_price - price
            sigma -= diff / v
            sigma = np.clip(sigma, 1e-6, 10.0)
            if abs(diff) < tol:
                return float(sigma)
        except Exception:
            break

    # ── Brent's method fallback ─────────────────────────────────────────────
    try:
        def objective(s):
            return black_scholes_price(S, K, T, r, s, option_type, q) - price

        iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter_brent)
        return float(iv)
    except Exception:
        return None


def implied_volatility_batch(options, r, q=0.0, n_jobs=1):
    """
    Solve IV for a list of option dicts with keys:
    price, S, K, T, option_type.

    Parameters
    ----------
    options : list[dict]
    r       : float  — Risk-free rate
    q       : float  — Dividend yield
    n_jobs  : int    — Parallelism (requires joblib if > 1)

    Returns
    -------
    ivs : list[float or None]
    """
    def _solve(opt):
        return implied_volatility(
            price=opt['price'],
            S=opt['S'],
            K=opt['K'],
            T=opt['T'],
            r=r,
            option_type=opt.get('option_type', 'call'),
            q=q,
        )

    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            return Parallel(n_jobs=n_jobs)(delayed(_solve)(o) for o in options)
        except ImportError:
            pass

    return [_solve(o) for o in options]