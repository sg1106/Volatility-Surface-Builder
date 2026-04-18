"""
main.py — Ultimate Options Analytics Dashboard
────────────────────────────────────────────────
Fetches live option chains, computes implied vols, Greeks, fits an
SVI + spline surface, runs arbitrage checks, and renders a rich
interactive Plotly HTML dashboard.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
from collections import defaultdict

from black_scholes import black_scholes_price, all_greeks
from iv_solver import implied_volatility
from surface_model import (
    VolatilitySurface, log_moneyness, check_put_call_parity,
    detect_butterfly_arb, detect_calendar_arb, svi_iv
)

# ─── Configuration ────────────────────────────────────────────────────────────

STOCK           = 'AAPL'
RISK_FREE_RATE  = 0.053
DIVIDEND_YIELD  = 0.005
MIN_VOLUME      = 10
MIN_OI          = 50
MAX_BID_ASK_PCT = 0.30
MAX_T_YEARS     = 2.0
MIN_T_DAYS      = 3
IV_RANGE        = (0.02, 3.0)
OUTPUT_HTML     = f'{STOCK}_vol_dashboard.html'

# ─── Colors ───────────────────────────────────────────────────────────────────

C = {
    'bg':      '#0d0d0f',
    'panel':   '#13131a',
    'border':  '#1e1e2e',
    'cyan':    '#00d4ff',
    'magenta': '#ff2d78',
    'gold':    '#ffd700',
    'green':   '#00ff88',
    'purple':  '#a855f7',
    'text':    '#e2e8f0',
    'muted':   '#64748b',
}

BASE_LAYOUT = dict(
    paper_bgcolor=C['bg'],
    plot_bgcolor=C['panel'],
    font=dict(family='monospace', color=C['text'], size=11),
    margin=dict(l=55, r=30, t=55, b=45),
)

AXIS_STYLE = dict(
    gridcolor=C['border'],
    zerolinecolor=C['border'],
    tickfont=dict(size=9),
    linecolor=C['border'],
)


# ─── Data fetching ────────────────────────────────────────────────────────────

def _safe_int(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        return int(val)
    except (TypeError, ValueError):
        return 0


def fetch_option_chain(ticker: str):
    stock = yf.Ticker(ticker)
    spot  = float(stock.history(period='5d')['Close'].dropna().iloc[-1])
    print(f'  Spot price : ${spot:.2f}')

    rows = []
    for exp in stock.options:
        try:
            chain = stock.option_chain(exp)
        except Exception as e:
            print(f'  [skip] {exp}: {e}')
            continue
        for otype, df in [('call', chain.calls), ('put', chain.puts)]:
            for _, r in df.iterrows():
                bid  = float(r.get('bid',  0) or 0)
                ask  = float(r.get('ask',  0) or 0)
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

    print(f'  Raw options: {len(rows)}')
    return rows, spot


def apply_quality_filters(option_data, spot, today):
    out = []
    for opt in option_data:
        T = (datetime.strptime(opt['expiration'], '%Y-%m-%d') - today).days / 365.0
        if T < MIN_T_DAYS / 365 or T > MAX_T_YEARS:
            continue
        if opt['mid'] <= 0 or opt['volume'] < MIN_VOLUME or opt['openInterest'] < MIN_OI:
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
            'iv':          iv,
            'moneyness':   log_moneyness(spot, opt['strike'], opt['T'], r, q),
            'model_price': black_scholes_price(spot, opt['strike'], opt['T'],
                                               r, iv, opt['option_type'], q),
            **{f'greek_{k}': v for k, v in g.items()},
        })
    return out


# ─── Arbitrage checks ─────────────────────────────────────────────────────────

def run_arb_checks(processed, spot, r, q=0.0):
    call_map = {(d['expiration'], d['strike']): d for d in processed if d['option_type'] == 'call'}
    put_map  = {(d['expiration'], d['strike']): d for d in processed if d['option_type'] == 'put'}

    pcp = []
    for key, c in call_map.items():
        if key in put_map:
            p   = put_map[key]
            chk = check_put_call_parity(c['mid'], p['mid'], spot, c['strike'], c['T'], r, q)
            if chk['violation']:
                pcp.append({**chk, 'strike': c['strike'], 'T': c['T']})

    slices = defaultdict(list)
    for d in processed:
        slices[round(d['T'], 4)].append({'strike': d['strike'], 'iv': d['iv'], 'T': d['T']})

    bf  = [v for sl in slices.values() for v in detect_butterfly_arb(sl)]
    cal = detect_calendar_arb(dict(slices))
    return {'pcp_violations': pcp, 'butterfly_violations': bf, 'calendar_violations': cal}


# ─── Figure 1: 3D Volatility Surface (standalone, no mixed axes) ──────────────

def make_surface_fig(processed, surface: VolatilitySurface, spot, ticker):
    calls = [d for d in processed if d['option_type'] == 'call']
    fig   = go.Figure()

    grid = surface.grid(n_strikes=60, n_maturities=30, strike_range=0.35)
    if grid:
        K_grid, T_grid, IV_grid = grid
        fig.add_trace(go.Surface(
            x=K_grid, y=T_grid, z=IV_grid,
            colorscale='Plasma', opacity=0.90, showscale=True,
            colorbar=dict(title='IV', tickformat='.0%', len=0.6),
            hovertemplate='Strike: $%{x:.0f}<br>Maturity: %{y:.2f}y<br>IV: %{z:.1%}<extra></extra>',
            name='Spline Surface',
        ))

    if calls:
        fig.add_trace(go.Scatter3d(
            x=[d['strike'] for d in calls],
            y=[d['T']      for d in calls],
            z=[d['iv']     for d in calls],
            mode='markers',
            marker=dict(size=2, color=C['cyan'], opacity=0.55),
            name='Observed IV',
            hovertemplate='K=$%{x:.0f}  T=%{y:.3f}y  IV=%{z:.2%}<extra></extra>',
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        height=550,
        title=dict(
            text=f'<b>{ticker} — Implied Volatility Surface (Calls)</b>',
            font=dict(size=16, color=C['text']), x=0.01,
        ),
        scene=dict(
            xaxis=dict(title='Strike ($)', backgroundcolor=C['panel'],
                       gridcolor=C['border'], showbackground=True),
            yaxis=dict(title='Maturity (yr)', backgroundcolor=C['panel'],
                       gridcolor=C['border'], showbackground=True),
            zaxis=dict(title='Implied Vol', backgroundcolor=C['panel'],
                       gridcolor=C['border'], showbackground=True, tickformat='.0%'),
            bgcolor=C['panel'],
        ),
        legend=dict(bgcolor=C['panel'], bordercolor=C['border'],
                    borderwidth=1, font=dict(size=9)),
    )
    return fig


# ─── Figure 2: 2D Analytics Grid (pure xy — no Surface traces) ───────────────

def make_analytics_fig(processed, surface: VolatilitySurface, spot, arb_results, ticker):
    calls     = [d for d in processed if d['option_type'] == 'call']
    puts      = [d for d in processed if d['option_type'] == 'put']
    today_str = datetime.now().strftime('%Y-%m-%d')
    palette   = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Vol Smile by Expiry (+ SVI fit)',
            'IV vs Log-Moneyness',
            'Open Interest by Strike',
            'Delta (Δ) vs Strike',
            'Gamma (Γ) vs Strike',
            'Vega (ν) vs Strike',
            'ATM IV Term Structure',
            'Theta (Θ) vs Strike',
            'Put-Call Parity Errors',
        ],
        vertical_spacing=0.11,
        horizontal_spacing=0.07,
    )

    # 1. Smile by expiry + SVI
    exps = sorted({d['expiration'] for d in calls})[:8]
    for i, exp in enumerate(exps):
        subset = sorted([d for d in calls if d['expiration'] == exp],
                        key=lambda x: x['strike'])
        if len(subset) < 3:
            continue
        T_val = subset[0]['T']
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=[d['moneyness'] for d in subset],
            y=[d['iv']        for d in subset],
            mode='markers', marker=dict(size=4, color=color, opacity=0.65),
            name=exp, legendgroup=exp,
        ), row=1, col=1)
        if T_val in surface.svi_slices:
            k_dense = np.linspace(
                min(d['moneyness'] for d in subset),
                max(d['moneyness'] for d in subset), 80)
            iv_svi = svi_iv(k_dense, surface.svi_slices[T_val])
            fig.add_trace(go.Scatter(
                x=k_dense, y=iv_svi, mode='lines',
                line=dict(color=color, width=1.5),
                name=f'{exp} SVI', legendgroup=exp, showlegend=False,
            ), row=1, col=1)

    # 2. IV all expiries scatter
    for otype, color, name in [('call', C['cyan'], 'Calls'), ('put', C['magenta'], 'Puts')]:
        sub = [d for d in processed if d['option_type'] == otype]
        fig.add_trace(go.Scatter(
            x=[d['moneyness'] for d in sub],
            y=[d['iv']        for d in sub],
            mode='markers', marker=dict(size=3, color=color, opacity=0.45),
            name=name,
        ), row=1, col=2)

    # 3. Open interest bars
    if calls:
        df_c = pd.DataFrame(calls).groupby('strike')['openInterest'].sum().reset_index()
        fig.add_trace(go.Bar(
            x=df_c['strike'], y=df_c['openInterest'],
            marker_color=C['cyan'], opacity=0.7, name='Call OI',
        ), row=1, col=3)
    if puts:
        df_p = pd.DataFrame(puts).groupby('strike')['openInterest'].sum().reset_index()
        fig.add_trace(go.Bar(
            x=df_p['strike'], y=df_p['openInterest'],
            marker_color=C['magenta'], opacity=0.7, name='Put OI',
        ), row=1, col=3)

    # Spot line — use add_shape (NOT add_vline, which breaks on mixed figs)
    fig.add_shape(type='line',
                  x0=spot, x1=spot, y0=0, y1=1,
                  xref='x3', yref='y3 domain',
                  line=dict(color=C['gold'], dash='dash', width=1.5))
    fig.add_annotation(x=spot, y=1, xref='x3', yref='y3 domain',
                       text='Spot', showarrow=False,
                       font=dict(color=C['gold'], size=9), yanchor='bottom')

    # 4. Delta
    for otype, color in [('call', C['cyan']), ('put', C['magenta'])]:
        sub = sorted([d for d in processed if d['option_type'] == otype],
                     key=lambda x: x['strike'])
        if sub:
            fig.add_trace(go.Scatter(
                x=[d['strike']      for d in sub],
                y=[d['greek_delta'] for d in sub],
                mode='markers', marker=dict(size=3, color=color, opacity=0.6),
                name=f'Δ {otype}',
            ), row=2, col=1)

    # 5. Gamma
    sub_c = sorted(calls, key=lambda x: x['strike'])
    if sub_c:
        fig.add_trace(go.Scatter(
            x=[d['strike']      for d in sub_c],
            y=[d['greek_gamma'] for d in sub_c],
            mode='markers', marker=dict(size=3, color=C['green'], opacity=0.6),
            name='Γ call',
        ), row=2, col=2)

    # 6. Vega
    if sub_c:
        fig.add_trace(go.Scatter(
            x=[d['strike']     for d in sub_c],
            y=[d['greek_vega'] for d in sub_c],
            mode='markers', marker=dict(size=3, color=C['purple'], opacity=0.6),
            name='ν call',
        ), row=2, col=3)

    # 7. ATM term structure
    atm_tol = 0.03
    for otype, color, name in [('call', C['cyan'], 'ATM Call'), ('put', C['magenta'], 'ATM Put')]:
        atm = [d for d in processed if d['option_type'] == otype and abs(d['moneyness']) < atm_tol]
        if atm:
            df_atm = pd.DataFrame(atm).groupby('T')['iv'].median().reset_index()
            fig.add_trace(go.Scatter(
                x=df_atm['T'], y=df_atm['iv'],
                mode='lines+markers', line=dict(color=color, width=2),
                marker=dict(size=5), name=name,
            ), row=3, col=1)

    svi_Ts  = sorted(surface.svi_slices.keys())
    svi_ivs = [float(svi_iv(0.0, surface.svi_slices[t])) for t in svi_Ts]
    if svi_Ts:
        fig.add_trace(go.Scatter(
            x=svi_Ts, y=svi_ivs, mode='markers',
            marker=dict(symbol='diamond', size=8, color=C['gold']),
            name='SVI ATM',
        ), row=3, col=1)

    # 8. Theta
    for otype, color in [('call', C['cyan']), ('put', C['magenta'])]:
        sub = sorted([d for d in processed if d['option_type'] == otype],
                     key=lambda x: x['strike'])
        if sub:
            fig.add_trace(go.Scatter(
                x=[d['strike']       for d in sub],
                y=[d['greek_theta']  for d in sub],
                mode='markers', marker=dict(size=3, color=color, opacity=0.6),
                name=f'Θ {otype}',
            ), row=3, col=2)

    # 9. PCP errors — use add_shape for zero line (NOT add_hline)
    pcp = arb_results.get('pcp_violations', [])
    if pcp:
        fig.add_trace(go.Scatter(
            x=[v['strike'] for v in pcp],
            y=[v['error']  for v in pcp],
            mode='markers', marker=dict(size=7, color=C['magenta'], symbol='x'),
            name='PCP Violation',
        ), row=3, col=3)
    else:
        fig.add_annotation(
            x=0.5, y=0.5, xref='x9 domain', yref='y9 domain',
            text='✓ No PCP violations', showarrow=False,
            font=dict(color=C['green'], size=12),
        )

    fig.add_shape(type='line',
                  x0=0, x1=1, y0=0, y1=0,
                  xref='x9 domain', yref='y9',
                  line=dict(color=C['muted'], dash='dot', width=1))

    # Axis styling
    for i in range(1, 10):
        fig.update_xaxes(AXIS_STYLE, row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)
        fig.update_yaxes(AXIS_STYLE, row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

    fig.update_yaxes(tickformat='.0%', row=1, col=1)
    fig.update_yaxes(tickformat='.0%', row=1, col=2)
    fig.update_yaxes(tickformat='.0%', row=3, col=1)

    fig.update_layout(
        **BASE_LAYOUT,
        height=920,
        title=dict(
            text=(f'<b>{ticker} — Options Analytics</b>'
                  f'  <span style="font-size:11px;color:{C["muted"]}">'
                  f'Spot ${spot:.2f} | r={RISK_FREE_RATE:.1%} | {today_str}</span>'),
            font=dict(size=15, color=C['text']), x=0.01,
        ),
        legend=dict(bgcolor=C['panel'], bordercolor=C['border'],
                    borderwidth=1, font=dict(size=9), x=1.01, y=1),
        barmode='overlay',
    )
    return fig


# ─── Figure 3: Summary Tables ─────────────────────────────────────────────────

def make_summary_fig(processed, surface, arb_results):
    calls = [d for d in processed if d['option_type'] == 'call']
    puts  = [d for d in processed if d['option_type'] == 'put']
    pcp_n = len(arb_results.get('pcp_violations',    []))
    bf_n  = len(arb_results.get('butterfly_violations', []))
    cal_n = len(arb_results.get('calendar_violations',  []))

    snap_metrics, snap_values = [], []
    atm_calls = sorted(calls, key=lambda x: abs(x['moneyness']))
    if atm_calls:
        a = atm_calls[0]
        snap_metrics = ['ATM Strike', 'ATM IV', 'Δ delta', 'Γ gamma',
                        'ν vega', 'Θ theta/day', 'Vanna', 'Volga']
        snap_values  = [
            f'${a["strike"]:.0f}',   f'{a["iv"]:.2%}',
            f'{a["greek_delta"]:+.4f}', f'{a["greek_gamma"]:.6f}',
            f'{a["greek_vega"]:.4f}',   f'{a["greek_theta"]:.4f}',
            f'{a["greek_vanna"]:.6f}',  f'{a["greek_volga"]:.6f}',
        ]

    stats_metrics = ['Total options', 'Calls', 'Puts',
                     'PCP violations', 'Butterfly arb', 'Calendar arb',
                     'SVI slices', 'Spline fitted']
    stats_values  = [str(len(processed)), str(len(calls)), str(len(puts)),
                     str(pcp_n), str(bf_n), str(cal_n),
                     str(len(surface.svi_slices)),
                     '✓' if surface._spline else '✗']

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'table'}, {'type': 'table'}]],
                        subplot_titles=['ATM Greeks Snapshot', 'Run Statistics'])

    hdr = dict(fill_color=C['border'], font=dict(color=C['text'], size=11),
               align='left', line_color=C['border'])

    fig.add_trace(go.Table(
        header=dict(**hdr, values=['<b>Greek</b>', '<b>Value</b>']),
        cells=dict(values=[snap_metrics, snap_values], fill_color=C['panel'],
                   font=dict(color=[C['text'], C['cyan']], size=11),
                   align='left', line_color=C['border']),
    ), row=1, col=1)

    fig.add_trace(go.Table(
        header=dict(**hdr, values=['<b>Metric</b>', '<b>Value</b>']),
        cells=dict(values=[stats_metrics, stats_values], fill_color=C['panel'],
                   font=dict(color=[C['text'], C['cyan']], size=11),
                   align='left', line_color=C['border']),
    ), row=1, col=2)

    fig.update_layout(**BASE_LAYOUT, height=320,
                      title=dict(text='<b>Summary</b>',
                                 font=dict(size=14, color=C['text']), x=0.01))
    return fig


# ─── Combine into one HTML ────────────────────────────────────────────────────

def write_dashboard(fig_surf, fig_2d, fig_tbl, spot, ticker, path):
    opts = dict(full_html=False, include_plotlyjs=False,
                config={'displayModeBar': True, 'scrollZoom': True})

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{ticker} — Vol Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *    {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {C['bg']}; font-family: 'Courier New', monospace;
            color: {C['text']}; padding: 20px; }}
    h1   {{ font-size: 22px; color: {C['cyan']}; letter-spacing: 2px;
            border-bottom: 1px solid {C['border']}; padding-bottom: 12px;
            margin-bottom: 8px; }}
    .sub {{ font-size: 12px; color: {C['muted']}; margin-bottom: 18px; }}
    .pnl {{ background: {C['panel']}; border: 1px solid {C['border']};
            border-radius: 6px; padding: 4px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>&#11041; {ticker} &mdash; Volatility Analytics Dashboard</h1>
  <div class="sub">
    Spot ${spot:.2f} &nbsp;|&nbsp; r={RISK_FREE_RATE:.1%}
    &nbsp;|&nbsp; q={DIVIDEND_YIELD:.1%}
    &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
  <div class="pnl">{pio.to_html(fig_surf, **opts)}</div>
  <div class="pnl">{pio.to_html(fig_2d,  **opts)}</div>
  <div class="pnl">{pio.to_html(fig_tbl, **opts)}</div>
</body>
</html>"""

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    print(f'\n{"═"*55}')
    print(f'  {STOCK} Options Analytics Dashboard')
    print(f'{"═"*55}')

    print('\n[1/5] Fetching option chain...')
    raw_data, spot = fetch_option_chain(STOCK)

    print('\n[2/5] Applying quality filters...')
    filtered = apply_quality_filters(raw_data, spot, today)
    print(f'  After filter : {len(filtered)} options')

    print('\n[3/5] Computing IVs & Greeks...')
    processed = compute_ivs(filtered, spot, RISK_FREE_RATE, DIVIDEND_YIELD)
    print(f'  Valid IVs    : {len(processed)}')

    if len(processed) < 20:
        print('  ⚠  Sparse — relaxing filters...')
        for opt in raw_data:
            opt.setdefault('T', (datetime.strptime(opt['expiration'], '%Y-%m-%d') - today).days / 365.0)
        relaxed   = [o for o in raw_data
                     if MIN_T_DAYS / 365 <= o.get('T', 0) <= MAX_T_YEARS and o['mid'] > 0]
        processed = compute_ivs(relaxed, spot, RISK_FREE_RATE, DIVIDEND_YIELD)
        print(f'  Valid IVs    : {len(processed)} (relaxed)')

    print('\n[4/5] Fitting volatility surface...')
    surface = VolatilitySurface()
    surface.fit(
        strikes    = [d['strike'] for d in processed],
        maturities = [d['T']      for d in processed],
        ivs        = [d['iv']     for d in processed],
        spot=spot, r=RISK_FREE_RATE, q=DIVIDEND_YIELD,
    )
    print(f'  SVI slices   : {len(surface.svi_slices)}')
    print(f'  Spline       : {"✓" if surface._spline else "✗"}')

    print('\n[5/5] Running arbitrage checks...')
    arb = run_arb_checks(processed, spot, RISK_FREE_RATE, DIVIDEND_YIELD)
    print(f'  PCP          : {len(arb["pcp_violations"])} violations')
    print(f'  Butterfly    : {len(arb["butterfly_violations"])} violations')
    print(f'  Calendar     : {len(arb["calendar_violations"])} violations')

    print('\n[*] Building dashboard...')
    fig_surf = make_surface_fig(processed, surface, spot, STOCK)
    fig_2d   = make_analytics_fig(processed, surface, spot, arb, STOCK)
    fig_tbl  = make_summary_fig(processed, surface, arb)
    write_dashboard(fig_surf, fig_2d, fig_tbl, spot, STOCK, OUTPUT_HTML)
    print(f'  ✓ Saved → {OUTPUT_HTML}')
    import webbrowser, os
    webbrowser.open(f'file://{os.path.abspath(OUTPUT_HTML)}')

    # Terminal snapshot
    atm_calls = sorted([d for d in processed if d['option_type'] == 'call'],
                       key=lambda x: abs(x['moneyness']))
    if atm_calls:
        a = atm_calls[0]
        print(f'\n{"─"*55}')
        print(f'  ATM Call  K=${a["strike"]:.0f}  T={a["T"]:.3f}y')
        print(f'{"─"*55}')
        for label, key, fmt in [
            ('IV',      'iv',           '.2%'),
            ('Δ delta', 'greek_delta',  '+.4f'),
            ('Γ gamma', 'greek_gamma',  '.6f'),
            ('ν vega',  'greek_vega',   '.4f'),
            ('Θ theta', 'greek_theta',  '.4f'),
            ('ρ rho',   'greek_rho',    '.4f'),
            ('Vanna',   'greek_vanna',  '.6f'),
            ('Volga',   'greek_volga',  '.6f'),
        ]:
            print(f'  {label:<10}: {a[key]:{fmt}}')
        print(f'{"─"*55}')

    print(f'\n  Open {OUTPUT_HTML} in your browser.\n')
    return processed, surface, arb


if __name__ == '__main__':
    processed, surface, arb = main()