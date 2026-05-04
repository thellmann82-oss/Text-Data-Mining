import json
import threading
import time
import uuid

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from genetic_classifier import GeneticRuleClassifier

app = Flask(__name__)
_jobs: dict = {}

FEATURE_NAMES = [
    'Tagesrendite',
    'Rendite (3T)',
    'Rendite (5T)',
    'MA5-Verhältnis',
    'MA10-Verhältnis',
    'MA20-Verhältnis',
    'Volumenänderung',
    'Hoch-Tief-Spanne',
    'Oberer Docht',
    'Unterer Docht',
]


def build_features(df: pd.DataFrame):
    df = df.copy()
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    open_ = df['Open'].squeeze()
    vol   = df['Volume'].squeeze()

    df['ret1']    = close.pct_change(1)
    df['ret3']    = close.pct_change(3)
    df['ret5']    = close.pct_change(5)
    df['ma5']     = close.rolling(5).mean()
    df['ma10']    = close.rolling(10).mean()
    df['ma20']    = close.rolling(20).mean()
    df['ma5r']    = close / df['ma5']
    df['ma10r']   = close / df['ma10']
    df['ma20r']   = close / df['ma20']
    df['vol_chg'] = vol.pct_change(1)

    hl = (high - low).replace(0, np.nan)
    df['hl_ratio'] = hl / close
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low  = pd.concat([open_, close], axis=1).min(axis=1)
    df['up_wick'] = (high - body_high) / hl
    df['lo_wick'] = (body_low - low)   / hl

    df['target'] = (close.shift(-1) > close).astype(int)
    df.dropna(inplace=True)

    feat_cols = ['ret1', 'ret3', 'ret5',
                 'ma5r', 'ma10r', 'ma20r',
                 'vol_chg', 'hl_ratio', 'up_wick', 'lo_wick']

    X      = df[feat_cols].values
    y      = df['target'].values
    dates  = df.index.strftime('%Y-%m-%d').tolist()
    prices = df['Close'].squeeze().round(2).tolist()
    return X, y, dates, prices


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stock')
def api_stock():
    ticker = request.args.get('ticker', 'AAPL').upper().strip()
    period = request.args.get('period', '1y')

    try:
        stock = yf.Ticker(ticker)
        df    = stock.history(period=period)
    except Exception as e:
        return jsonify({'fehler': str(e)}), 500

    if df.empty:
        return jsonify({'fehler': f'Keine Daten für "{ticker}" gefunden.'}), 404

    X, y, dates, prices = build_features(df)

    name, currency, sector = ticker, 'USD', ''
    try:
        info     = stock.info
        name     = info.get('longName', ticker)
        currency = info.get('currency', 'USD')
        sector   = info.get('sector', '')
    except Exception:
        pass

    return jsonify({
        'ticker':    ticker,
        'name':      name,
        'currency':  currency,
        'sektor':    sector,
        'anzahl':    len(X),
        'daten':     dates,
        'kurse':     prices,
        'up_anteil': round(float(y.mean()) * 100, 1),
        'merkmale':  FEATURE_NAMES,
    })


@app.route('/api/train', methods=['POST'])
def api_train():
    data        = request.json or {}
    ticker      = data.get('ticker', 'AAPL').upper().strip()
    period      = data.get('period', '1y')
    generations = min(int(data.get('generations', 30)), 100)
    pop_size    = min(int(data.get('population', 50)), 200)

    try:
        df = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        return jsonify({'fehler': str(e)}), 500

    if df.empty:
        return jsonify({'fehler': 'Keine Daten gefunden.'}), 404

    X, y, _, _ = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {'status': 'running', 'progress': [], 'result': None, 'clf': None}

    def run():
        clf = GeneticRuleClassifier(population_size=pop_size, generations=generations)

        def cb(gen, total, score):
            _jobs[job_id]['progress'].append(
                {'gen': gen, 'total': total, 'score': round(score, 4)}
            )

        clf.fit(X_train, y_train, feature_names=FEATURE_NAMES, progress_callback=cb)

        train_acc = float(accuracy_score(y_train, clf.predict(X_train)))
        test_acc  = float(accuracy_score(y_test,  clf.predict(X_test)))

        _jobs[job_id].update({
            'status': 'done',
            'clf':    clf,
            'result': {
                'train_genauigkeit': round(train_acc, 4),
                'test_genauigkeit':  round(test_acc, 4),
                'fitness_verlauf':   clf.history,
                'regeln':            clf.get_rules(),
                'anzahl_regeln':     len(clf.best_model.rules),
            },
        })

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/api/progress/<job_id>')
def api_progress(job_id):
    def generate():
        sent = 0
        while True:
            job = _jobs.get(job_id)
            if job is None:
                yield f"data: {json.dumps({'fehler': 'Unbekannter Job'})}\n\n"
                return

            new_items = job['progress'][sent:]
            for item in new_items:
                yield f"data: {json.dumps(item)}\n\n"
            sent += len(new_items)

            if job['status'] == 'done':
                payload = {**job['result'], 'fertig': True}
                yield f"data: {json.dumps(payload)}\n\n"
                return

            time.sleep(0.05)

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/network/<job_id>')
def api_network(job_id):
    job = _jobs.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'fehler': 'Modell nicht bereit.'}), 400

    clf   = job['clf']
    rules = clf.best_model.rules
    n     = clf._n_features
    names = clf.feature_names

    node_count = [0] * n
    class_up   = [0] * n
    edge_count = [[0] * n for _ in range(n)]

    for rule in rules:
        feats = list({f for f, _, _ in rule.conditions})
        for f in feats:
            node_count[f] += 1
            if rule.label == 1:
                class_up[f] += 1
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                edge_count[feats[i]][feats[j]] += 1
                edge_count[feats[j]][feats[i]] += 1

    nodes = [
        {
            'id':       i,
            'name':     names[i],
            'count':    node_count[i],
            'up_ratio': round(class_up[i] / node_count[i], 2) if node_count[i] else 0.5,
        }
        for i in range(n) if node_count[i] > 0
    ]

    edges = [
        {'source': i, 'target': j, 'weight': edge_count[i][j]}
        for i in range(n)
        for j in range(i + 1, n)
        if edge_count[i][j] > 0
    ]

    return jsonify({'nodes': nodes, 'edges': edges, 'n_regeln': len(rules)})


@app.route('/api/vorhersage/<job_id>')
def api_vorhersage(job_id):
    job = _jobs.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'fehler': 'Modell nicht bereit.'}), 400

    ticker = request.args.get('ticker', 'AAPL').upper().strip()

    try:
        df = yf.Ticker(ticker).history(period='3mo')
    except Exception as e:
        return jsonify({'fehler': str(e)}), 500

    if df.empty:
        return jsonify({'fehler': 'Keine Daten gefunden.'}), 404

    clf = job['clf']
    X, y, dates, prices = build_features(df)
    preds = clf.predict(X)
    last  = int(preds[-1])

    return jsonify({
        'daten':                  dates[-30:],
        'kurse':                  prices[-30:],
        'vorhersagen':            preds[-30:].tolist(),
        'tatsaechlich':           y[-30:].tolist(),
        'letzte_vorhersage':      last,
        'letzte_vorhersage_text': 'Steigt ↑' if last == 1 else 'Fällt ↓',
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
