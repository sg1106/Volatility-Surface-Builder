"""
analytics.py — Quant computation pipeline (the actual "backend logic").

Wraps black_scholes.py / iv_solver.py / surface_model.py with live data
fetching and produces a single JSON-serializable payload consumed by the
HTML/CSS/JS frontend over the Flask API in app.py.

Nothing in this file renders anything — it only computes.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict

from black_scholes import all_greeks
from iv_solver import implied_volatility
from surface_model import (
    VolatilitySurface, log_moneyness, check_put_call_parity,
    detect_butterfly_arb, detect_calendar_arb, svi_iv,
)

# ─── Tunables ─────────────────────────────────────────────────────────────────

MIN_VOLUME      = 10
MIN_OI          = 50
MAX_BID_ASK_PCT = 0.30
MAX_T_YEARS     = 2.0
MIN_T_DAYS      = 3
IV_RANGE        = (0.02, 3.0)
ATM_TOLERANCE   = 0.03   # log-moneyness band considered "ATM"
MAX_EXPIRATIONS = 10     # cap network round-trips per request (each is a separate Yahoo call)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_int(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        return int(val)
    except (TypeError, ValueError):
        return 0


# ─── Data fetching ────────────────────────────────────────────────────────────

def fetch_option_chain(ticker: str):
    """Pull spot price and the full option chain (all expiries) for a ticker."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5d')
    if hist.empty:
        raise ValueError(f'No price data found for ticker "{ticker}".')
    spot = float(hist['Close'].dropna().iloc[-1])

    expirations = stock.options
    if not expirations:
        raise ValueError(f'No listed options chain found for "{ticker}".')
    # yfinance returns expirations in chronological order; keep only the
    # nearest N so one request doesn't make 15-20+ sequential Yahoo calls.
    expirations = expirations[:MAX_EXPIRATIONS]

    rows = []
    for exp in expirations:
        try:
            chain = stock.option_chain(exp)
        except Exception:
            continue
        for otype, df in [('call', chain.calls), ('put', chain.puts)]:
            for _, r in df.iterrows():
                bid  = float(r.get('bid', 0) or 0)
                ask  = float(r.get('ask', 0) or 0)
                last = float(r.get('lastPrice', 0) or 0)
                mid  = (bid + ask) / 2 if (bid + ask) > 0 else last
                rows.append({
                    'expiration':   exp,
                    'strike':       float(r['strike']),
                    'bid':          bid,
                    'ask':          ask,
                    'mid':          mid,
                    'lastPrice':    last,
                    'volume':       _safe_int(r.get('volume')),
                    'openInterest': _safe_int(r.get('openInterest')),
                    'option_type':  otype,
                })

    return rows, spot


def apply_quality_filters(option_data, today, relax=False):
    """Drop illiquid / stale / wide-spread contracts. `relax=True` keeps
    everything except basic sanity (price>0, valid maturity window)."""
    out = []
    for opt in option_data:
        T = (datetime.strptime(opt['expiration'], '%Y-%m-%d') - today).days / 365.0
        if T < MIN_T_DAYS / 365 or T > MAX_T_YEARS:
            continue
        if opt['mid'] <= 0:
            continue
        if not relax:
            if opt['volume'] < MIN_VOLUME or opt['openInterest'] < MIN_OI:
                continue
            spread = opt['ask'] - opt['bid']
            if opt['mid'] > 0 and spread / opt['mid'] > MAX_BID_ASK_PCT:
                continue
        opt['T'] = T
        out.append(opt)
    return out


# ─── IV + Greeks ──────────────────────────────────────────────────────────────

def compute_ivs(filtered, spot, r, q=0.0):
    out = []
    for opt in filtered:
        iv = implied_volatility(
            price=opt['mid'], S=spot, K=opt['strike'],
            T=opt['T'], r=r, option_type=opt['option_type'], q=q,
        )
        if iv is None or not (IV_RANGE[0] < iv < IV_RANGE[1]):
            continue
        g = all_greeks(spot, opt['strike'], opt['T'], r, iv, opt['option_type'], q)
        out.append({
            **opt,
            'iv':        float(iv),
            'moneyness': float(log_moneyness(spot, opt['strike'], opt['T'], r, q)),
            'greeks':    {k: float(v) for k, v in g.items()},
        })
    return out


# ─── Arbitrage diagnostics ────────────────────────────────────────────────────

def run_arb_checks(processed, spot, r, q=0.0):
    call_map = {(d['expiration'], d['strike']): d for d in processed if d['option_type'] == 'call'}
    put_map  = {(d['expiration'], d['strike']): d for d in processed if d['option_type'] == 'put'}

    pcp = []
    for key, c in call_map.items():
        if key in put_map:
            p   = put_map[key]
            chk = check_put_call_parity(c['mid'], p['mid'], spot, c['strike'], c['T'], r, q)
            if chk['violation']:
                pcp.append({
                    'strike':      c['strike'],
                    'T':           c['T'],
                    'theoretical': float(chk['theoretical']),
                    'actual':      float(chk['actual']),
                    'error':       float(chk['error']),
                })

    slices = defaultdict(list)
    for d in processed:
        slices[round(d['T'], 4)].append({'strike': d['strike'], 'iv': d['iv'], 'T': d['T']})

    bf = [
        {'strike': v['strike'], 'iv': float(v['iv']), 'd2w': float(v['d2w'])}
        for sl in slices.values() for v in detect_butterfly_arb(sl)
    ]
    cal = [
        {'strike': v['strike'], 'T1': v['T1'], 'T2': v['T2'],
         'w1': float(v['w1']), 'w2': float(v['w2'])}
        for v in detect_calendar_arb(dict(slices))
    ]

    return {'pcp_violations': pcp, 'butterfly_violations': bf, 'calendar_violations': cal}


# ─── ATM term structure ───────────────────────────────────────────────────────

def _atm_series(processed, option_type):
    pts = [d for d in processed if d['option_type'] == option_type and abs(d['moneyness']) < ATM_TOLERANCE]
    if not pts:
        return []
    df = pd.DataFrame(pts).groupby('T')['iv'].median().reset_index()
    return [{'T': float(t), 'iv': float(v)} for t, v in zip(df['T'], df['iv'])]


# ─── Orchestration ────────────────────────────────────────────────────────────

def run_analysis(ticker: str, r: float, q: float = 0.0):
    """
    Full pipeline: fetch → filter → IV/Greeks → surface fit → arb checks.
    Returns a single JSON-serializable dict for the frontend.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    raw, spot = fetch_option_chain(ticker)

    filtered  = apply_quality_filters(raw, today)
    processed = compute_ivs(filtered, spot, r, q)

    relaxed_used = False
    if len(processed) < 20:
        relaxed_used = True
        filtered  = apply_quality_filters(raw, today, relax=True)
        processed = compute_ivs(filtered, spot, r, q)

    if not processed:
        raise ValueError(
            f'No valid options with a computable implied volatility for "{ticker}". '
            'The chain may be too illiquid or the inputs out of range.'
        )

    surface = VolatilitySurface()
    surface.fit(
        strikes    = [d['strike'] for d in processed],
        maturities = [d['T']      for d in processed],
        ivs        = [d['iv']     for d in processed],
        spot=spot, r=r, q=q,
    )

    arb = run_arb_checks(processed, spot, r, q)

    # 3D surface grid
    surface_payload = None
    grid = surface.grid(n_strikes=60, n_maturities=30, strike_range=0.35)
    if grid is not None:
        K_grid, T_grid, IV_grid = grid
        surface_payload = {
            'x': K_grid[0, :].tolist(),
            'y': T_grid[:, 0].tolist(),
            'z': IV_grid.tolist(),
        }

    # Dense SVI curves per expiry (for smile overlay), in log-moneyness space
    svi_curves = {}
    for T, params in surface.svi_slices.items():
        ks = [d['moneyness'] for d in processed if abs(d['T'] - T) < 1e-6]
        if not ks:
            continue
        k_dense  = np.linspace(min(ks), max(ks), 60)
        iv_dense = svi_iv(k_dense, params)
        svi_curves[f'{T:.6f}'] = {'k': k_dense.tolist(), 'iv': iv_dense.tolist()}

    svi_atm = sorted(
        [{'T': float(T), 'iv': float(svi_iv(0.0, p))} for T, p in surface.svi_slices.items()],
        key=lambda x: x['T'],
    )

    atm_term_structure = {
        'call': _atm_series(processed, 'call'),
        'put':  _atm_series(processed, 'put'),
        'svi':  svi_atm,
    }

    atm_calls = sorted(
        [d for d in processed if d['option_type'] == 'call'],
        key=lambda x: abs(x['moneyness']),
    )
    atm_snapshot = None
    if atm_calls:
        a = atm_calls[0]
        atm_snapshot = {
            'strike': a['strike'], 'expiration': a['expiration'],
            'T': a['T'], 'iv': a['iv'], **a['greeks'],
        }

    calls_n = sum(1 for d in processed if d['option_type'] == 'call')
    puts_n  = len(processed) - calls_n

    options_payload = [{
        'expiration':   d['expiration'],
        'strike':       d['strike'],
        'option_type':  d['option_type'],
        'mid':          d['mid'],
        'iv':           d['iv'],
        'T':            d['T'],
        'moneyness':    d['moneyness'],
        'volume':       d['volume'],
        'openInterest': d['openInterest'],
        'greeks':       d['greeks'],
    } for d in processed]

    return {
        'meta': {
            'ticker':           ticker,
            'spot':             spot,
            'risk_free_rate':   r,
            'dividend_yield':   q,
            'timestamp':        datetime.now().isoformat(),
            'total_options':    len(processed),
            'calls':            calls_n,
            'puts':             puts_n,
            'svi_slices':       len(surface.svi_slices),
            'spline_fitted':    surface._spline is not None,
            'relaxed_filters':  relaxed_used,
        },
        'options':              options_payload,
        'surface':               surface_payload,
        'svi_curves':            svi_curves,
        'atm_term_structure':    atm_term_structure,
        'arbitrage':              arb,
        'atm_snapshot':          atm_snapshot,
    }