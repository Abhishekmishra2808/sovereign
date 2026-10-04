# LISWET1 final closure — CUTEst-derived c headline

The headline result below uses `c_cutest = -grad_obj(0)` directly from
PyCUTEst. The parser-derived c run remains a secondary comparison.
The sign follows the SIF linear term `-c'x`: for
`0.5*x'x-c'x+constant`, `grad_obj(0)=-c`.

## CUTEst-derived c headline

- `||c_cutest-c_parsed||inf = 4.994205049513312e-11`
- Largest difference: one-based index 1804 (zero-based 1803),
  `c_parsed=1.015610921450058`, `c_cutest=1.0156109215`,
  difference `4.994205049513312e-11`
- Full top-10 index list is in `c_source_comparison.json`.
- `c_cutest` was written from `-grad_obj(0)` at 17 significant digits; the
  direct PyCUTEst check gives `||grad_obj(0)+c_cutest||inf = 0`.

Sovereign solved `c_cutest.txt` with status `OPTIMAL`:

- Runtime: `0.6736082001589239` seconds
- GI iterations: `2576/40040`
- Drops: `288`
- Certificate: primal `4.3368086899420177e-19`, dual `0`,
  stationarity `9.26681095140408e-12`, complementarity
  `3.0842599105736523e-15`
- Independent mpmath distance objective:
  `7.22189498657455311187386580409`
- PyCUTEst `p.obj(x)`:
  `7.221894982162432`
- Raw objective difference:
  `4.4121211118738658e-9`

The raw difference was investigated, not dismissed. PyCUTEst reports
`p.obj(0)=505.53197247`, while `0.5*c_cutest'c_cutest` is
`505.5319724744125`, a difference of `4.4125e-9`. After subtracting that
stored SIF objective constant, the PyCUTEst objective increment and the
mpmath QP objective differ by only `4.78e-13`. Thus the remaining raw gap is
in the SIFDecode-stored objective constant, not in the c vector or x solve.

The CUTEst-derived run has constraint range
`[-2.220446049250313e-16, 2.220446049250313e-16]`. Its independent mpmath
certificate reports primal violation `4.0e-16`, minimum multiplier
`0.1388170684250183954`, stationarity `2.9969e-13`, and complementarity
`1.4077521411074005e-12`.

The CUTEst-derived and parser-derived primal solutions differ by only
`||x_cutest-x_parsed||inf = 8.033573806187633e-13`, at one-based index 2002.

## Instance identity

The downloaded official file is `benchmarks/data/LISWET1.SIF`. Its active
parameters are `N=2000`, `K=2`, giving 2000 constraints and 2002 variables.
Only this active option was run.

Commented-out options observed in the SIF and not run:

- `N=100, K=3/4/5/6` → 103/104/105/106 variables;
- `N=400, K=2/3` → 402/403 variables;
- `N=2000, K=1` → 2001 variables;
- `N=10000, K=1/2` → 10001/10002 variables.

The `N=10000` and `N=20000` experiments elsewhere in the repository are
synthetic-scalability experiments, not this official LISWET1 run.

## Parser-derived c secondary result

The final `x.txt` and `c.txt` use 17-significant-digit decimal output.

- Status: `OPTIMAL`
- Solver runtime: `0.8996882999781519` seconds
- GI iterations: `2576/40040`
- Drops: `288`
- Active constraints: `2000`
- Guard trips / refreshes: `0 / 0`
- QP objective `0.5*x'x-c'x`: `-498.310077487419846`
- C++ certificate: primal `3.2526065174565133e-19`, dual `0`,
  stationarity `6.16239698159049e-12`, complementarity `3.17561923654096e-15`

Independent 50-digit verification from the same files:

- Primal violation: `3.0e-16`
- Minimum multiplier: `0.1388170684218540984`
- Stationarity: `7.171216e-13`
- Complementarity: `1.4012352525409948e-12`
- QP objective: `-498.310077487419833843544385152`
- Distance objective: `0.5*||x-c||² = 7.22189498656168399875891696926`

## PyCUTEst confirmation for parser-derived c

PyCUTEst evaluated the same final `x.txt`:

- `p.n = 2002`, `p.m = 2000`
- `p.obj(x) = 7.221894982161224`
- `||c_cutest-c.txt||inf = 4.994205049513312e-11`
- `min(p.cons(x)) = -2.220446049250313e-16`
- `max(p.cons(x)) = 2.220446049250313e-16`

The PyCUTEst/multiprecision objective difference is
`4.400459998758917e-9` (relative `6.093220694772189e-10`), above `1e-11`.
This is not evidence of a wrong `c`: both use the same final `x.txt` bytes;
PyCUTEst reconstructs `c` from the SIF while mpmath reads decimal `c.txt`.
The large difference between the QP objective and the SIF objective is the
constant `0.5*||c||² = 505.5319724739815178`; after applying that conversion,
the remaining `4.4e-9` is double-precision evaluation/order plus the
`4.99e-11` SIF-generated-gradient versus rounded-`c.txt` difference across
2002 terms. The PyCUTEst feasibility result and the independent certificate
provide the separate optimality checks.

## Independent solver

CVXOPT sparse primal QP cross-check:

- `||x_GI-x_ref||inf = 3.338248203976768e-7`
- Relative distance-objective difference:
  `1.0399420597806804e-9`
- Reference primal infeasibility: `4.697312813494839e-14`
- CVXOPT status: `unknown`; its dual residual is poorly scaled
  (`1.1691342190255524e-5`) for this second-difference system, so this is
  reported as a cross-check, not as the certificate.

## SHA-256

- `LISWET1.SIF`:
  `6384C64C5C6049F851E5C2E7ACB8112185F1F82C321349B5FEAD537E69499F9C`
- `c.txt`:
  `323051C8BA406D7B57E76DF2A7DAA642467A2999CA1EAF75E2633D5169441332`
- `x.txt`:
  `F6A69711BDDB0B8E6E36209AB3D06073399C81DF276AC3B4F3DBCC1044DF9942`
- `c_cutest.txt`:
  `5DB3F3CB2B08F9CB03B1E8E3A56799407B02C8178FFA01FDF30D1D88311C962A`
- `x_cutest.txt`:
  `4C68AD114B8FB375AB7E94762DE002B688D361E8C5E19191CC966BFD1D3DCD54`

## Exact commands

```powershell
New-Item -ItemType Directory -Force benchmarks\data | Out-Null
Invoke-WebRequest `
  -Uri "https://bitbucket.org/optrove/sif/raw/master/LISWET1.SIF" `
  -OutFile "benchmarks\data\LISWET1.SIF"

python benchmarks/tools/run_liswet1.py `
  --sif benchmarks/data/LISWET1.SIF `
  --binary build64/solver/sovereign.exe `
  --out-dir benchmarks/reports/liswet_real

python benchmarks/tools/verify_liswet_independent.py `
  --c-file benchmarks/reports/liswet_real/c.txt `
  --x-file benchmarks/reports/liswet_real/x.txt `
  --out benchmarks/reports/liswet_real/independent.json

python benchmarks/tools/crosscheck_liswet_real.py `
  --c-file benchmarks/reports/liswet_real/c.txt `
  --x-file benchmarks/reports/liswet_real/x.txt `
  --out benchmarks/reports/liswet_real/cvxopt.json

wsl.exe -- bash -lc `
  "docker run --rm -e DUMP_CUTEST=1 `
   -e LISWET_CUTEST_OUT=/workspace/benchmarks/reports/liswet_real/c_cutest.txt `
   -v /mnt/c/Users/abhis/Desktop/26119:/workspace `
   continuumio/miniconda3 bash `
   /workspace/benchmarks/tools/run_pycutest_container.sh"

python benchmarks/tools/compare_liswet_c_sources.py `
  --parsed benchmarks/reports/liswet_real/c.txt `
  --cutest benchmarks/reports/liswet_real/c_cutest.txt `
  --out benchmarks/reports/liswet_real/c_source_comparison.json

python benchmarks/tools/run_liswet1.py `
  --c-file benchmarks/reports/liswet_real/c_cutest.txt `
  --expected-length 2002 `
  --binary build64/solver/sovereign.exe `
  --out-dir benchmarks/reports/liswet_real_cutest

python benchmarks/tools/verify_liswet_independent.py `
  --c-file benchmarks/reports/liswet_real_cutest/c.txt `
  --x-file benchmarks/reports/liswet_real_cutest/x.txt `
  --out benchmarks/reports/liswet_real_cutest/independent.json

python benchmarks/tools/compare_liswet_x_sources.py `
  --parsed-x benchmarks/reports/liswet_real/x.txt `
  --cutest-x benchmarks/reports/liswet_real_cutest/x.txt `
  --out benchmarks/reports/liswet_real_cutest/x_source_comparison.json

wsl.exe -- bash -lc `
  "docker run --rm `
   -e LISWET_X_FILE=/workspace/benchmarks/reports/liswet_real_cutest/x.txt `
   -e LISWET_C_FILE=/workspace/benchmarks/reports/liswet_real/c_cutest.txt `
   -v /mnt/c/Users/abhis/Desktop/26119:/workspace `
   continuumio/miniconda3 bash `
   /workspace/benchmarks/tools/run_pycutest_container.sh"

Get-FileHash benchmarks\data\LISWET1.SIF -Algorithm SHA256
Get-FileHash benchmarks\reports\liswet_real\c.txt -Algorithm SHA256
Get-FileHash benchmarks\reports\liswet_real\x.txt -Algorithm SHA256
```
