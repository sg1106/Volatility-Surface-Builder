"""
app.py — Flask backend.

Serves:
  • GET /                  → static/index.html  (the HTML/CSS/JS frontend)
  • GET /api/analyze        → runs the full quant pipeline, returns JSON
  • GET /api/health          → liveness check

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, jsonify, request, send_from_directory
import numpy as np

from analytics import run_analysis

app = Flask(__name__, static_folder='static', static_url_path='')


def _sanitize(obj):
    """Recursively replace NaN/Inf with None so the JSON payload is
    strictly valid (Python's json module otherwise emits literal NaN,
    which breaks JS's JSON.parse)."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/analyze')
def analyze():
    ticker = request.args.get('ticker', 'AAPL').strip().upper()
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required.'}), 400

    try:
        r = float(request.args.get('r', 0.053))
        q = float(request.args.get('q', 0.005))
    except ValueError:
        return jsonify({'error': 'Risk-free rate and dividend yield must be numbers.'}), 400

    try:
        result = run_analysis(ticker, r, q)
        return jsonify(_sanitize(result))
    except ValueError as e:
        # Expected, user-facing errors (bad ticker, no chain, etc.)
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Unexpected server error: {e}'}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print('\n' + '═' * 55)
    print('  Volatility Terrain Survey — backend running')
    print('═' * 55)
    print(f'  Open http://127.0.0.1:{port} in your browser\n')
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', host='0.0.0.0', port=port)
