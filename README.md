# Volatility Terrain Survey

A live options-analytics dashboard: Flask backend doing all the quant work
(Black-Scholes, Newton-Raphson/Brent implied vol, SVI surface fitting,
arbitrage diagnostics), with a plain HTML/CSS/JS frontend (Plotly.js for
charts) — no Streamlit, no frontend framework.

## Project layout

```
volatility-dashboard/
├── app.py              Flask app — serves the API + the static frontend
├── analytics.py         Data pipeline: fetch chain → filter → IV/Greeks →
│                        surface fit → arbitrage checks → JSON payload
├── black_scholes.py    Pricing + full Greeks (delta, gamma, vega, theta,
│                        rho, vanna, volga, charm, speed)
├── iv_solver.py        Newton-Raphson with Brent's-method fallback
├── surface_model.py    SVI parametrization, spline surface, arb checks
├── requirements.txt
└── static/
    ├── index.html       Page structure
    ├── style.css         Design system
    └── app.js            Fetches /api/analyze and renders every chart
```

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. The page loads,
auto-runs a survey for AAPL, and you can change the ticker / rate /
dividend yield and hit "Survey" to re-run.

## API

`GET /api/analyze?ticker=AAPL&r=0.053&q=0.005`

Runs the full pipeline server-side and returns one JSON payload containing:
processed options with Greeks, the fitted surface grid, dense SVI curves
per expiry, the ATM term structure, and all arbitrage flags. The frontend
does zero computation — it only renders what the backend sends.

`GET /api/health` — liveness check.

## Notes

- All quant logic lives in `analytics.py` / `black_scholes.py` /
  `iv_solver.py` / `surface_model.py` — completely framework-agnostic
  pure Python, importable from anywhere (a script, a notebook, a
  different web framework) without modification.
- If a chain is too illiquid at the default volume/OI thresholds, the
  backend automatically relaxes filters and flags this in the response
  (`meta.relaxed_filters`), surfaced in the UI as a banner.
- The 3D surface and 2D charts are rendered with Plotly.js loaded from a
  CDN — swap in your own chart library by editing `static/app.js` only;
  the backend contract doesn't need to change.
