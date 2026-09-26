# Benchmarks (external comparison only)

HiGHS / other solvers are imported **only** in this folder. Production `solver/` never links them.

## Quick start

```powershell
pip install highspy
python benchmarks/tools/generate_datasets.py
python benchmarks/runners/run_benchmarks.py --suite all --timeout 180
```

Outputs:
- `benchmarks/reports/latest.csv`
- `benchmarks/reports/EVIDENCE.md`

## Suites

| Suite | Contents |
|-------|----------|
| `smoke` | `examples/models` toys |
| `netlib` | Netlib **AFIRO** (MPS → JSON) |
| `miplib` | Synthetic knapsack ablation (not official MIPLIB IDs) |
| `robustness` | Kuhn degeneracy, ill-conditioned coeffs, weak LP relaxation MILP |
| `scale` | Transportation LP 20×20 (400 vars) and 50×50 (2500 vars) |
| `all` | All of the above |

Official **MIPLIB 2017** instances live under `datasets/miplib/official/` and are
benchmarked separately:

```powershell
python benchmarks/runners/run_miplib_official.py
```

That writes `reports/miplib_official.json` (LP-relax matches vs HiGHS; full MILP
timeouts reported honestly). Evidence §2b distinguishes official vs synthetic.

## Tools

- `tools/mps_to_json.py` — free-format MPS → sovereign JSON
- `tools/generate_datasets.py` — regenerate robustness / scale / synthetic knapsack JSON

## Pitch note

Lead with **AFIRO vs HiGHS**, official MIPLIB LP-relax matches, and named robustness cases.
QP default is Mehrotra IPM (Frank–Wolfe is a labeled fallback). Full MIPLIB MILP
competitiveness is still open.
