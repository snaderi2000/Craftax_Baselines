"""Serve the offline atlas and optional CPU k-means fitting on localhost."""
from __future__ import annotations

import os
# Keep a local experiment from taking every CPU on the shared research host.
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Lock
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from concept_mapping.reward_transition_explorer.build import (
    DEFAULT_OUT, METHODS, assignment, diagnostics, eligible, fit_kmeans, normalize, validate_bank)


class Experiment:
    def __init__(self, out):
        self.out = Path(out).resolve()
        manifest = json.loads((self.out / 'manifest.json').read_text())
        self.source = Path(manifest['source'])
        self.indices = np.asarray(manifest['indices'], np.int32)
        self.bank = None
        self.lock = Lock()

    def fit(self, params):
        if not isinstance(params, dict):
            raise ValueError('Expected a JSON object')
        method, mode = params.get('method'), params.get('mode')
        if method not in METHODS or mode != 'achievement_reward':
            raise ValueError('Unknown representation or reward filter')
        bounds = dict(k=(2, 256), seed=(0, 1000000), cap=(256, 100000), iterations=(5, 100))
        for name, (low, high) in bounds.items():
            value = params.get(name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer between {low} and {high}')
        k, seed, cap, iterations = (params[n] for n in ('k', 'seed', 'cap', 'iterations'))
        key = f'{mode}_{method}_k{k}_seed{seed}_cap{cap}_iter{iterations}'
        cache = self.out / 'fits' / (key + '.json')
        if cache.exists():
            return json.loads(cache.read_text())
        if self.bank is None:
            with np.load(self.source / 'transition_bank.npz', allow_pickle=False) as data:
                self.bank = dict(data)
            validate_bank(self.bank)
        bank = self.bank
        mask = eligible(bank['rewards'], mode)
        tr = np.flatnonzero(mask & (bank['split_id'] == 0))
        te = np.flatnonzero(mask & (bank['split_id'] == 1))
        if len(tr) < k or not len(te):
            raise ValueError('Insufficient reference or held-out transitions for this fit')
        with np.load(self.source / 'features.npz', allow_pickle=False) as features:
            x = normalize(features[method])
        rng = np.random.default_rng(seed)
        fit_indices = rng.choice(tr, min(cap, len(tr)), replace=False)
        print(f'Fitting {key} on {len(fit_indices):,} reference transitions', flush=True)
        centers = fit_kmeans(x[fit_indices], k, seed, iterations)
        a, b = assignment(x[tr], centers), assignment(x[te], centers)
        all_assign = np.full(len(mask), -1, np.int16)
        all_assign[tr], all_assign[te] = a, b
        display = all_assign[self.indices]
        valid = display >= 0
        distance = np.full(len(display), -1., np.float32)
        distance[valid] = ((x[self.indices[valid]] - centers[display[valid]]) ** 2).sum(1)
        config = dict(id=key, method=method, mode=mode, k=k, seed=seed, iterations=iterations,
                      fit_n=len(fit_indices), reference_n=len(tr), test_n=len(te), cached=False,
                      assignments=display.tolist(), distances=np.round(distance, 6).tolist(),
                      **diagnostics(bank, tr, te, a, b, k))
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(config, separators=(',', ':'), allow_nan=False))
        # Preserve original feature-space centroids and exact fitting provenance.
        np.savez_compressed(cache.with_suffix('.npz'), centers=centers, fit_indices=fit_indices,
                            reference_indices=tr, test_indices=te,
                            reference_assignments=a, test_assignments=b)
        print(f'Saved {key}', flush=True)
        return config


def handler_for(experiment):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, data, kind):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                self.send(200, (experiment.out / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif path == '/api/health':
                self.send(200, b'{"ok":true}', 'application/json')
            else:
                self.send(404, b'Not found', 'text/plain')

        def do_POST(self):
            if self.path != '/api/fit':
                self.send(404, b'Not found', 'text/plain'); return
            # A browser opened on another origin cannot trigger local compute.
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                self.send(403, b'Use the explorer on this local server', 'text/plain'); return
            if not experiment.lock.acquire(blocking=False):
                self.send(409, b'A fit is already running. Try again after it finishes.', 'text/plain'); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size')
                params = json.loads(self.rfile.read(length))
                result = experiment.fit(params)
                self.send(200, json.dumps(result, allow_nan=False).encode(), 'application/json')
            except (ValueError, KeyError) as exc:
                self.send(400, str(exc).encode(), 'text/plain')
            except Exception as exc:
                print(f'Fit error: {exc}', file=sys.stderr, flush=True)
                self.send(500, b'Fit failed; see the server log for details.', 'text/plain')
            finally:
                experiment.lock.release()
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    experiment = Experiment(args.out)
    if not (experiment.out / 'index.html').is_file():
        parser.error('Build the explorer first with build.py')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(experiment))
    print(f'Explorer: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
