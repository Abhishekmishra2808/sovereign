"""Bounded release smoke test: launches only its own server and always cleans up."""
import argparse
import json
import math
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir', type=Path, default=ROOT / 'build-session/solver/Release')
    parser.add_argument('--static', type=Path, default=ROOT / 'api/static')
    parser.add_argument('--models', type=Path, default=ROOT / 'examples/models')
    args = parser.parse_args()
    proc = subprocess.Popen([str(args.bin_dir.resolve() / 'sovereign-server.exe'), '--static',
        str(args.static.resolve()), '--model-root', str(args.models.resolve()), '--threads', '2'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    lines = queue.Queue()
    threading.Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True).start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        line = lines.get(timeout=15)
        assert line.startswith('SOVEREIGN_PORT='), line
        base = 'http://127.0.0.1:' + line.strip().split('=')[1]

        def request(path, data=None, headers=None):
            req = urllib.request.Request(base + path,
                data=json.dumps(data).encode() if data is not None else None,
                headers={'Content-Type': 'application/json', **(headers or {})})
            with opener.open(req, timeout=10) as response:
                return response.read()

        for _ in range(30):
            try:
                health = json.loads(request('/api/health'))
                break
            except urllib.error.URLError:
                time.sleep(.1)
        assert health['status'] == 'ok'
        assert json.loads(request('/api/system'))['capabilities']['MPS']
        assert json.loads(request('/api/problems'))['models']
        assert b'<html' in request('/dashboard/').lower()
        assert json.loads(request('/dashboard/data/benchmarks.json'))['rows']

        mps = 'OBJSENSE MAX\nROWS\n N OBJ\n L CAP\nCOLUMNS\n M0 \'MARKER\' \'INTORG\'\n X OBJ 5 CAP 1\n Y OBJ 4 CAP 1\n M1 \'MARKER\' \'INTEND\'\nRHS\n R CAP 1 OBJ -2\nENDATA\n'
        summary = json.loads(request('/api/validate', {'modelJson': mps, 'modelFormat': 'mps'}))
        assert summary['valid'] and summary['integer_vars'] == 2, summary
        assert not json.loads(request('/api/validate', {'modelJson': '{oops'}))['valid']
        bad = {'problem_type':'LP','variables':[{'name':'x'}], 'objective':{'linear':{'unknown':1}}}
        assert not json.loads(request('/api/validate', {'modelJson':json.dumps(bad)}))['valid']

        def solve(body, expected):
            job_id = json.loads(request('/api/jobs', body))['jobId']
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                result = json.loads(request('/api/jobs/' + job_id))
                if result['state'] in ('COMPLETED', 'FAILED', 'CANCELLED'):
                    break
                time.sleep(.05)
            assert result['state'] == 'COMPLETED', result
            assert result['solver_status'] == 'OPTIMAL', result
            assert math.isclose(result['objective'], expected, rel_tol=1e-9, abs_tol=1e-9), result
            assert result['verification']['is_valid'], result
            export = json.loads(request('/api/jobs/' + job_id + '/solution'))
            assert export['solution'] == result['solution']
            assert export['objective'] == result['objective']
            stream = request('/api/jobs/' + job_id + '/events').decode()
            assert 'COMPLETED' in stream, stream
            return result

        solve({'modelJson': mps, 'modelFormat':'mps'}, 7)
        solve({'modelPath':'sample_milp.json'}, 8)
        # JSON escape round trips and full double precision in both snapshot/export.
        name = 'x"\\\n\U0001f680'
        model = {'problem_type':'LP', 'variables':[{'name':name, 'lower_bound':1/3, 'upper_bound':1/3}],
                 'objective':{'linear':{name:1}}, 'constraints':[]}
        result = solve({'modelJson': json.dumps(model), 'presolve':True}, 1/3)
        assert name in result['solution']
        for path, body, headers, expected in [
            ('/api/jobs', {'modelPath':'../outside.json'}, {}, 400),
            ('/api/jobs', {'modelPath':'sample_lp.json'}, {'Origin':'https://evil.example'}, 403),
            ('/api/health', None, {'Host':'evil.example'}, 421),
        ]:
            try:
                request(path, body, headers)
                raise AssertionError('unsafe request was accepted')
            except urllib.error.HTTPError as error:
                assert error.code == expected, (error.code, expected)
        print('PASS: API, MPS/JSON solve, validation, precision, SSE, export, offline assets, Host/Origin/path guards')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        proc.stdout.close()


if __name__ == '__main__':
    main()
