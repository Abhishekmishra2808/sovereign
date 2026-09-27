"""Compare every bundled MPS to independent JSON fixtures; flag known reference loss."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=ROOT / 'build-session/solver/Release/sovereign.exe')
    args = parser.parse_args()
    results = []
    for path in sorted((ROOT / 'benchmarks/datasets').rglob('*.mps')):
        process = subprocess.run([str(args.exe.resolve()), 'convert', str(path)], capture_output=True, text=True, timeout=30)
        assert process.returncode == 0, (path.name, process.stderr)
        model = json.loads(process.stdout)
        reference = json.loads(path.with_suffix('.json').read_text())
        assert model['problem_type'] == reference['problem_type'], path.name
        assert model['sense'] == reference['sense'], path.name
        def variables(m):
            return {v['name']:(v['type']=='continuous',v.get('lower_bound',0),v.get('upper_bound') or (0 if v.get('upper_bound')==0 else 1e30)) for v in m['variables']}
        assert variables(model) == variables(reference), path.name
        assert model['objective']['linear'] == reference['objective']['linear'], path.name
        def constraints(m):
            return {c['name']:(c['sense'],c['rhs'],c['linear']) for c in m['constraints']}
        actual, expected = constraints(model), constraints(reference)
        if path.name == 'rentacar.mps':
            # Python reference drops RANGES: D###14 has width 1000, D###24 width 500.
            # Verify precisely those intervals and every unaffected row.
            for row, width in [('D###14',1000),('D###24',500)]:
                old = expected.pop(row)
                hi, lo = actual.pop(row), actual.pop(row+'_lo')
                assert hi[0]=='<=' and lo[0]=='>=' and hi[1]-lo[1]==width
                assert hi[2]==lo[2]==old[2]
                assert old[1] in (hi[1],lo[1])
            extra = set(actual) - set(expected)
            assert all(actual[name] == ('=', 0.0, {}) for name in extra), extra
            for name in extra:
                actual.pop(name)
            assert actual == expected
            verdict = 'PASS; reference omits two range upper bounds and empty tautologies (independently checked)'
        else:
            assert actual == expected, path.name
            verdict = 'PASS; matches independent JSON'
        results.append({'file':str(path.relative_to(ROOT)), 'rows':len(model['constraints']),
                        'columns':len(model['variables']), 'result':verdict})
        print(path.name + ': ' + verdict)
    target = ROOT / 'benchmarks/reports/mps_audit.json'
    target.write_text(json.dumps(results, indent=2)+'\n')
    print(f'PASS: {len(results)} MPS files parsed and checked')


if __name__ == '__main__':
    main()
