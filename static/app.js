/* ════════════════════════════════════════════════════════════════════
   app.js — Volatility Terrain Survey frontend
   Talks to the Flask backend (/api/analyze) and renders every chart
   with Plotly.js. No framework — vanilla DOM + fetch.
   ════════════════════════════════════════════════════════════════════ */

const C = {
  ink:    '#0a0e1a',
  panel:  '#111a2c',
  panel2: '#0d1422',
  border: '#28344a',
  brass:  '#c9a227',
  tide:   '#3f93a0',
  coral:  '#e2553d',
  chalk:  '#e9e4d6',
  muted:  '#6f7890',
};

const FONT_MONO = 'IBM Plex Mono, monospace';

const LOADING_LINES = [
  'Casting the line…',
  'Reading the depths…',
  'Tracing the contours…',
  'Charting strike by maturity…',
  'Fitting the SVI slices…',
  'Squaring the parity book…',
];

let loadingTimer = null;

// ─── DOM refs ──────────────────────────────────────────────────────────────

const els = {
  form:          document.getElementById('controlsForm'),
  ticker:        document.getElementById('tickerInput'),
  rate:          document.getElementById('rateInput'),
  div:           document.getElementById('divInput'),
  btn:           document.getElementById('surveyBtn'),
  overlay:       document.getElementById('loadingOverlay'),
  loadingText:   document.getElementById('loadingText'),
  errorBanner:   document.getElementById('errorBanner'),
  errorMessage:  document.getElementById('errorMessage'),
  relaxNotice:   document.getElementById('relaxNotice'),
  roSpot:        document.getElementById('roSpot'),
  roContracts:   document.getElementById('roContracts'),
  roSvi:         document.getElementById('roSvi'),
  roArb:         document.getElementById('roArb'),
  roTime:        document.getElementById('roTime'),
  atmTable:      document.querySelector('#atmTable tbody'),
  statsTable:    document.querySelector('#statsTable tbody'),
  arbList:       document.getElementById('arbList'),
};

// ─── Format helpers ────────────────────────────────────────────────────────

const fmtPct  = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : `${(v * 100).toFixed(d)}%`;
const fmtNum  = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : v.toFixed(d);
const fmtUsd  = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : `$${v.toFixed(d)}`;
const fmtSign = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;

// ─── Shared Plotly layout base ─────────────────────────────────────────────

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: C.panel,
    plot_bgcolor:  C.panel,
    font:          { family: FONT_MONO, color: C.chalk, size: 10.5 },
    margin:        { l: 48, r: 16, t: 8, b: 38 },
    showlegend:    true,
    legend:        { bgcolor: C.panel, bordercolor: C.border, borderwidth: 1, font: { size: 9 } },
    xaxis:         { gridcolor: C.border, zerolinecolor: C.border, linecolor: C.border },
    yaxis:         { gridcolor: C.border, zerolinecolor: C.border, linecolor: C.border },
  }, extra);
}

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

// ─── Loading overlay ───────────────────────────────────────────────────────

function showLoading() {
  els.overlay.hidden = false;
  els.btn.disabled = true;
  let i = 0;
  els.loadingText.textContent = LOADING_LINES[0];
  loadingTimer = setInterval(() => {
    i = (i + 1) % LOADING_LINES.length;
    els.loadingText.textContent = LOADING_LINES[i];
  }, 1100);
}

function hideLoading() {
  els.overlay.hidden = true;
  els.btn.disabled = false;
  clearInterval(loadingTimer);
}

function showError(message) {
  els.errorBanner.hidden = false;
  els.errorMessage.textContent = message;
}

function clearBanners() {
  els.errorBanner.hidden = true;
  els.relaxNotice.hidden = true;
}

// ─── Fetch + orchestrate ────────────────────────────────────────────────────

async function runSurvey(e) {
  if (e) e.preventDefault();
  clearBanners();
  showLoading();

  const ticker = els.ticker.value.trim().toUpperCase();
  const r = parseFloat(els.rate.value) / 100;
  const q = parseFloat(els.div.value) / 100;

  if (!ticker) {
    hideLoading();
    showError('Enter a ticker symbol.');
    return;
  }

  try {
    const url = `/api/analyze?ticker=${encodeURIComponent(ticker)}&r=${r}&q=${q}`;
    const res = await fetch(url);
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `Request failed (${res.status}).`);
    }

    renderDashboard(data);
  } catch (err) {
    showError(err.message || 'Could not reach the backend. Is app.py running?');
  } finally {
    hideLoading();
  }
}

// ─── Render: readout strip ──────────────────────────────────────────────────

function renderReadout(meta, arb) {
  els.roSpot.textContent = fmtUsd(meta.spot);
  els.roContracts.textContent = `${meta.total_options} (${meta.calls}C / ${meta.puts}P)`;
  els.roSvi.textContent = `${meta.svi_slices} slice${meta.svi_slices === 1 ? '' : 's'}`;
  const arbN = arb.pcp_violations.length + arb.butterfly_violations.length + arb.calendar_violations.length;
  els.roArb.textContent = arbN === 0 ? 'none' : `${arbN} flagged`;
  els.roArb.style.color = arbN === 0 ? C.tide : C.coral;
  const t = new Date(meta.timestamp);
  els.roTime.textContent = t.toLocaleString(undefined, { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' });

  if (meta.relaxed_filters) els.relaxNotice.hidden = false;
}

// ─── Render: 3D surface (hero) ───────────────────────────────────────────────

function renderSurface(data) {
  const traces = [];

  if (data.surface) {
    traces.push({
      type: 'surface',
      x: data.surface.x, y: data.surface.y, z: data.surface.z,
      colorscale: [[0, C.ink], [0.5, C.tide], [1, C.brass]],
      opacity: 0.92,
      showscale: true,
      colorbar: { title: 'IV', tickformat: '.0%', len: 0.62, outlinecolor: C.border, tickfont: { size: 9 } },
      hovertemplate: 'Strike $%{x:.0f}<br>Maturity %{y:.2f}y<br>IV %{z:.1%}<extra></extra>',
      name: 'Fitted surface',
      contours: { z: { show: true, usecolormap: true, highlightcolor: C.chalk, project: { z: true } } },
    });
  }

  const calls = data.options.filter(d => d.option_type === 'call');
  traces.push({
    type: 'scatter3d', mode: 'markers',
    x: calls.map(d => d.strike), y: calls.map(d => d.T), z: calls.map(d => d.iv),
    marker: { size: 2, color: C.chalk, opacity: 0.5 },
    name: 'Observed IV',
    hovertemplate: 'K=$%{x:.0f}  T=%{y:.3f}y  IV=%{z:.2%}<extra></extra>',
  });

  const layout = baseLayout({
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: { title: 'Strike ($)', gridcolor: C.border, backgroundcolor: C.panel, showbackground: true, color: C.muted },
      yaxis: { title: 'Maturity (yr)', gridcolor: C.border, backgroundcolor: C.panel, showbackground: true, color: C.muted },
      zaxis: { title: 'Implied vol', tickformat: '.0%', gridcolor: C.border, backgroundcolor: C.panel, showbackground: true, color: C.muted },
      bgcolor: C.panel,
      camera: { eye: { x: 1.5, y: -1.5, z: 0.7 } },
    },
  });

  Plotly.newPlot('surface3d', traces, layout, PLOT_CONFIG);
}

// ─── Render: smile by expiry + SVI overlay ──────────────────────────────────

const PALETTE = [C.brass, C.tide, '#a36bd1', '#5fb567', '#d18a4c', '#5f8fd1', '#c95f8f', '#9fb55f'];

function renderSmile(data) {
  const calls = data.options.filter(d => d.option_type === 'call');
  const byExp = groupBy(calls, d => d.expiration);
  const exps = Object.keys(byExp).sort().slice(0, 8);

  const traces = [];
  exps.forEach((exp, i) => {
    const pts = byExp[exp].sort((a, b) => a.strike - b.strike);
    if (pts.length < 3) return;
    const color = PALETTE[i % PALETTE.length];

    traces.push({
      type: 'scatter', mode: 'markers',
      x: pts.map(d => d.moneyness), y: pts.map(d => d.iv),
      marker: { size: 5, color, opacity: 0.7 },
      name: exp, legendgroup: exp,
    });

    const T = pts[0].T;
    const svi = data.svi_curves[T.toFixed(6)];
    if (svi) {
      traces.push({
        type: 'scatter', mode: 'lines',
        x: svi.k, y: svi.iv,
        line: { color, width: 1.5 },
        name: `${exp} SVI`, legendgroup: exp, showlegend: false,
      });
    }
  });

  Plotly.newPlot('chartSmile', traces, baseLayout({
    yaxis: { tickformat: '.0%', gridcolor: C.border, zerolinecolor: C.border },
    legend: { bgcolor: C.panel, font: { size: 8 }, orientation: 'h', y: -0.25 },
  }), PLOT_CONFIG);
}

// ─── Render: IV vs moneyness scatter ─────────────────────────────────────────

function renderIvScatter(data) {
  const calls = data.options.filter(d => d.option_type === 'call');
  const puts  = data.options.filter(d => d.option_type === 'put');

  const traces = [
    { type: 'scatter', mode: 'markers', x: calls.map(d => d.moneyness), y: calls.map(d => d.iv),
      marker: { size: 3, color: C.brass, opacity: 0.45 }, name: 'Calls' },
    { type: 'scatter', mode: 'markers', x: puts.map(d => d.moneyness), y: puts.map(d => d.iv),
      marker: { size: 3, color: C.tide, opacity: 0.45 }, name: 'Puts' },
  ];

  Plotly.newPlot('chartIvScatter', traces, baseLayout({
    yaxis: { tickformat: '.0%', gridcolor: C.border, zerolinecolor: C.border },
  }), PLOT_CONFIG);
}

// ─── Render: open interest bars ──────────────────────────────────────────────

function renderOI(data) {
  const calls = data.options.filter(d => d.option_type === 'call');
  const puts  = data.options.filter(d => d.option_type === 'put');

  const callOi = sumByKey(calls, 'strike', 'openInterest');
  const putOi  = sumByKey(puts,  'strike', 'openInterest');

  const traces = [
    { type: 'bar', x: Object.keys(callOi).map(Number), y: Object.values(callOi),
      marker: { color: C.brass, opacity: 0.75 }, name: 'Call OI' },
    { type: 'bar', x: Object.keys(putOi).map(Number), y: Object.values(putOi),
      marker: { color: C.tide, opacity: 0.75 }, name: 'Put OI' },
  ];

  const layout = baseLayout({ barmode: 'overlay' });
  layout.shapes = [{
    type: 'line', x0: data.meta.spot, x1: data.meta.spot, y0: 0, y1: 1,
    xref: 'x', yref: 'paper',
    line: { color: C.brass, dash: 'dash', width: 1.4 },
  }];
  layout.annotations = [{
    x: data.meta.spot, y: 1, xref: 'x', yref: 'paper',
    text: 'Spot', showarrow: false, font: { color: C.brass, size: 9 }, yanchor: 'bottom',
  }];

  Plotly.newPlot('chartOI', traces, layout, PLOT_CONFIG);
}

// ─── Render: Greek scatter (generic helper) ──────────────────────────────────

function renderGreekScatter(elId, data, greekKey, { bothSides = true, single = false } = {}) {
  const traces = [];
  if (single) {
    const calls = data.options.filter(d => d.option_type === 'call').sort((a, b) => a.strike - b.strike);
    traces.push({
      type: 'scatter', mode: 'markers',
      x: calls.map(d => d.strike), y: calls.map(d => d.greeks[greekKey]),
      marker: { size: 3, color: C.brass, opacity: 0.6 }, name: `${greekKey} (call)`,
    });
  } else {
    const calls = data.options.filter(d => d.option_type === 'call').sort((a, b) => a.strike - b.strike);
    const puts  = data.options.filter(d => d.option_type === 'put').sort((a, b) => a.strike - b.strike);
    traces.push({
      type: 'scatter', mode: 'markers',
      x: calls.map(d => d.strike), y: calls.map(d => d.greeks[greekKey]),
      marker: { size: 3, color: C.brass, opacity: 0.6 }, name: `${greekKey} call`,
    });
    if (bothSides) {
      traces.push({
        type: 'scatter', mode: 'markers',
        x: puts.map(d => d.strike), y: puts.map(d => d.greeks[greekKey]),
        marker: { size: 3, color: C.tide, opacity: 0.6 }, name: `${greekKey} put`,
      });
    }
  }
  Plotly.newPlot(elId, traces, baseLayout(), PLOT_CONFIG);
}

// ─── Render: ATM term structure ──────────────────────────────────────────────

function renderAtmTerm(data) {
  const ts = data.atm_term_structure;
  const traces = [];

  if (ts.call.length) {
    traces.push({
      type: 'scatter', mode: 'lines+markers',
      x: ts.call.map(d => d.T), y: ts.call.map(d => d.iv),
      line: { color: C.brass, width: 2 }, marker: { size: 5 }, name: 'ATM call',
    });
  }
  if (ts.put.length) {
    traces.push({
      type: 'scatter', mode: 'lines+markers',
      x: ts.put.map(d => d.T), y: ts.put.map(d => d.iv),
      line: { color: C.tide, width: 2, dash: 'dot' }, marker: { size: 5 }, name: 'ATM put',
    });
  }
  if (ts.svi.length) {
    traces.push({
      type: 'scatter', mode: 'markers',
      x: ts.svi.map(d => d.T), y: ts.svi.map(d => d.iv),
      marker: { symbol: 'diamond', size: 8, color: '#f0d27a' }, name: 'SVI ATM',
    });
  }

  Plotly.newPlot('chartAtmTerm', traces, baseLayout({
    yaxis: { tickformat: '.0%', gridcolor: C.border, zerolinecolor: C.border },
  }), PLOT_CONFIG);
}

// ─── Render: PCP errors ───────────────────────────────────────────────────────

function renderPCP(data) {
  const pcp = data.arbitrage.pcp_violations;
  const traces = [];

  if (pcp.length) {
    traces.push({
      type: 'scatter', mode: 'markers',
      x: pcp.map(v => v.strike), y: pcp.map(v => v.error),
      marker: { size: 8, color: C.coral, symbol: 'x' }, name: 'PCP violation',
    });
  }

  const layout = baseLayout();
  layout.shapes = [{
    type: 'line', x0: 0, x1: 1, y0: 0, y1: 0,
    xref: 'paper', yref: 'y',
    line: { color: C.muted, dash: 'dot', width: 1 },
  }];
  if (!pcp.length) {
    layout.annotations = [{
      x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
      text: '✓ no PCP violations', showarrow: false,
      font: { color: C.tide, size: 12 },
    }];
  }

  Plotly.newPlot('chartPCP', traces, layout, PLOT_CONFIG);
}

// ─── Render: ship's log tables ────────────────────────────────────────────────

function renderAtmTable(snap) {
  els.atmTable.innerHTML = '';
  if (!snap) {
    els.atmTable.innerHTML = '<tr><td colspan="2">No ATM call found</td></tr>';
    return;
  }
  const rows = [
    ['Strike', fmtUsd(snap.strike, 0)],
    ['Expiration', snap.expiration],
    ['Implied vol', fmtPct(snap.iv)],
    ['Delta Δ', fmtSign(snap.delta)],
    ['Gamma Γ', fmtNum(snap.gamma, 6)],
    ['Vega ν', fmtNum(snap.vega)],
    ['Theta Θ /day', fmtNum(snap.theta)],
    ['Rho ρ', fmtNum(snap.rho)],
    ['Vanna', fmtNum(snap.vanna, 6)],
    ['Volga', fmtNum(snap.volga, 6)],
  ];
  for (const [label, val] of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${label}</td><td>${val}</td>`;
    els.atmTable.appendChild(tr);
  }
}

function renderStatsTable(meta) {
  els.statsTable.innerHTML = '';
  const rows = [
    ['Total options', meta.total_options],
    ['Calls', meta.calls],
    ['Puts', meta.puts],
    ['SVI slices fitted', meta.svi_slices],
    ['Spline fitted', meta.spline_fitted ? '✓' : '✗'],
    ['Filters relaxed', meta.relaxed_filters ? 'yes' : 'no'],
    ['Risk-free rate', fmtPct(meta.risk_free_rate, 2)],
    ['Dividend yield', fmtPct(meta.dividend_yield, 2)],
  ];
  for (const [label, val] of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${label}</td><td>${val}</td>`;
    els.statsTable.appendChild(tr);
  }
}

function renderArbList(arb) {
  els.arbList.innerHTML = '';
  const total = arb.pcp_violations.length + arb.butterfly_violations.length + arb.calendar_violations.length;

  if (total === 0) {
    els.arbList.innerHTML = '<div class="arblist__empty">✓ No arbitrage violations detected</div>';
    return;
  }

  arb.pcp_violations.slice(0, 8).forEach(v => {
    addArbRow(`Parity · K=$${v.strike.toFixed(0)} T=${v.T.toFixed(2)}y`, `err ${v.error.toFixed(2)}`);
  });
  arb.butterfly_violations.slice(0, 8).forEach(v => {
    addArbRow(`Butterfly · K=$${v.strike.toFixed(0)}`, `d²w ${v.d2w.toFixed(4)}`);
  });
  arb.calendar_violations.slice(0, 8).forEach(v => {
    addArbRow(`Calendar · K=$${v.strike.toFixed(0)} ${v.T1.toFixed(2)}→${v.T2.toFixed(2)}y`, `Δw ${(v.w2 - v.w1).toFixed(4)}`);
  });
}

function addArbRow(label, value) {
  const row = document.createElement('div');
  row.className = 'arblist__row';
  row.innerHTML = `<span>${label}</span><span>${value}</span>`;
  els.arbList.appendChild(row);
}

// ─── Small utilities ───────────────────────────────────────────────────────

function groupBy(arr, keyFn) {
  return arr.reduce((acc, item) => {
    const k = keyFn(item);
    (acc[k] = acc[k] || []).push(item);
    return acc;
  }, {});
}

function sumByKey(arr, groupKey, sumKey) {
  return arr.reduce((acc, item) => {
    const k = item[groupKey];
    acc[k] = (acc[k] || 0) + (item[sumKey] || 0);
    return acc;
  }, {});
}

// ─── Top-level render ──────────────────────────────────────────────────────

function renderDashboard(data) {
  renderReadout(data.meta, data.arbitrage);
  renderSurface(data);
  renderSmile(data);
  renderIvScatter(data);
  renderOI(data);
  renderGreekScatter('chartDelta', data, 'delta', { bothSides: true });
  renderGreekScatter('chartGamma', data, 'gamma', { single: true });
  renderGreekScatter('chartVega',  data, 'vega',  { single: true });
  renderAtmTerm(data);
  renderGreekScatter('chartTheta', data, 'theta', { bothSides: true });
  renderPCP(data);
  renderAtmTable(data.atm_snapshot);
  renderStatsTable(data.meta);
  renderArbList(data.arbitrage);
}

// ─── Init ──────────────────────────────────────────────────────────────────

els.form.addEventListener('submit', runSurvey);

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => runSurvey());
} else {
  runSurvey();
}
