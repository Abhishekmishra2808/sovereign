# Stage 2 MIPLIB 2017 MILP sweep

This report contains only official MIPLIB instances from the checked-in manifest. Synthetic fixtures are not included.

- Selection rule: Historical HiGHS outcome SOLVED within 300 seconds and max(rows, columns) <= 3000.
- Instances with complete inputs and results: 30/30
- Time limit: 600.0 s per solver
- Threads: Sovereign 1, HiGHS 1
- Commit: `420744cc57b5b4c8246340d70413bf25b96074ff`
- Working tree dirty at run time: `True`
- HiGHS: `1.15.1`
- Independent feasibility tolerance: `1e-06`
- Independent integrality tolerance: `1e-05`
- MIP relative-gap tolerance: `1e-06`

## Summary

- Sovereign verified solved: **10/30**
- HiGHS verified solved: **30/30**
- Sovereign shifted geometric mean, 10 s shift: **235.754 s**
- HiGHS shifted geometric mean, 10 s shift: **19.6489 s**
- Performance profile data: `performance_profile.csv`
- Performance profile plot: `performance_profile.png`

## Missing inputs

- None; all manifest inputs were materialized.

## Instance provenance

- `10teams`: rows=230, cols=2025, integers=1800, nnz=12150, bytes=536783, sha256=`cc5d351ee5e56c64b2931d2205afcd4463e93d587abe546856cf601894b6ec43`, reference=923.9999999999997
- `b-ball`: rows=30, cols=100, integers=88, nnz=209, bytes=15239, sha256=`b0d21fd470e519128ac6003ea3f3a46625d7b987494532c5bfcc10dacd04d0a6`, reference=-1.50000000000001
- `beasleyC3`: rows=1750, cols=2500, integers=1250, nnz=5000, bytes=262195, sha256=`728c9616ca793393f90d6bf7780b4871e71f33add9f81eacfc12bf05976130c8`, reference=753.9999999999128
- `blend2`: rows=274, cols=353, integers=264, nnz=1409, bytes=63421, sha256=`48c009606d5e949baa8e5e060c8b0cafd3eb96f9d615d98fff0d626d5b6e503f`, reference=7.598985
- `dcmulti`: rows=290, cols=548, integers=75, nnz=1315, bytes=76205, sha256=`a5f5147f5450d1b954e2299569f340191529bd07bdb349996dbe21576f1e7665`, reference=188182.0
- `enlight_hard`: rows=100, cols=200, integers=200, nnz=560, bytes=56971, sha256=`572ca23c17d0ad734895e8338af458525a753ee76bdb117d2917e4069c6b65b0`, reference=37.0
- `exp-1-500-5-5`: rows=550, cols=990, integers=250, nnz=1980, bytes=209869, sha256=`32bcb5e23d511b3ee530e7016765cc724e06bc77225648669c443aa7be0a68f2`, reference=65887.0
- `fiber`: rows=363, cols=1298, integers=1254, nnz=2944, bytes=183540, sha256=`75d2d56cb3559940b554c4ceda3225513d4f8648a6629996de2b0db6422181bd`, reference=405935.18
- `flugpl`: rows=18, cols=18, integers=11, nnz=46, bytes=4161, sha256=`a1f0cb79a95639450dd984473a00efca24365457153d7fcb859c6581ddab4c2c`, reference=1201500.0
- `gt2`: rows=29, cols=188, integers=188, nnz=376, bytes=23771, sha256=`3ddd85dc12f8460a92b36b1424a00b563170d56ee615340dec037d7a875e9050`, reference=21165.99999999978
- `khb05250`: rows=101, cols=1350, integers=24, nnz=2700, bytes=135500, sha256=`261c4498faab02e5de68aca63b08a17b7243985e2a979c9d505dba4fa0c86555`, reference=106940225.9999999
- `markshare_4_0`: rows=4, cols=34, integers=30, nnz=123, bytes=5377, sha256=`a05d1c8c1a1e0646ea31965fd5e05af7a41801db3cb1820b2ab4cd95c7bc32db`, reference=1.0
- `mas76`: rows=12, cols=151, integers=150, nnz=1640, bytes=62523, sha256=`114b73dbe17c1d1b4d37a09311424571781225e01edfbec1c6acb320214593f6`, reference=40005.05398999999
- `mik-250-20-75-4`: rows=195, cols=270, integers=250, nnz=9270, bytes=283262, sha256=`b29fe88b56bf3c924894f473cee77e25efb4dbaccb322e95c98b9210fb875e6d`, reference=-52301.0
- `misc07`: rows=212, cols=260, integers=259, nnz=8619, bytes=286262, sha256=`20cb89af2d0ca7ded2dad43817e620893c6b189bd66018d2be3317ba596aa615`, reference=2810.0
- `mod010`: rows=146, cols=2655, integers=2655, nnz=11203, bytes=544964, sha256=`0a2a0888816fd25e31ff4512ca8974272dacf387854cd450e94946332d89325d`, reference=6548.0
- `neos-911970`: rows=107, cols=888, integers=840, nnz=3408, bytes=118807, sha256=`31e8f835a9499de4897eb23838453cb83bee093abb8fab881b44f9937236eaa6`, reference=54.76
- `neos17`: rows=486, cols=535, integers=300, nnz=4931, bytes=977794, sha256=`4ea347bc9061a49b4737f2e77b3c15541c031eb8df14881cc9a4bd96a2d65c00`, reference=0.1500025774
- `neos5`: rows=63, cols=63, integers=53, nnz=2016, bytes=69767, sha256=`6004edda3e48c0d2e6b109e88c059bd874875535fa420cd486461e68a7375870`, reference=15.0
- `noswot`: rows=182, cols=128, integers=100, nnz=735, bytes=31909, sha256=`1f26832d33f0cb9eea1b5eb512c0db766fac1ba0802eb2d214c8d3358e77949a`, reference=-41.00000885
- `p0201`: rows=133, cols=201, integers=201, nnz=1923, bytes=78610, sha256=`f9bd3802c05f506cdba111051baaff6a62288c00f4292b02722303663e8e2392`, reference=7614.999999999997
- `p200x1188c`: rows=1388, cols=2376, integers=1188, nnz=4752, bytes=288222, sha256=`52e616e7a1e0dc95b14a8e381bda45bd1618cadcf5f8393609b1b7fe81a5459e`, reference=15078.0
- `pg`: rows=125, cols=2700, integers=100, nnz=5200, bytes=362964, sha256=`765a365c6d851573ef12a4c6c08b5fec0c3514e195986618c9d37075e84a3a0c`, reference=-8674.34260712
- `qiu`: rows=1192, cols=840, integers=48, nnz=3432, bytes=137711, sha256=`86990b99a5c702a14b2ac2ea4f46eaa88027f150d6d3e6ec8c8edc3b8c88d58a`, reference=-132.87313695
- `qnet1`: rows=503, cols=1541, integers=1417, nnz=4622, bytes=198064, sha256=`eaa0737929f199b168bd30510dd7ab59ed2ef6fb04d5ac05ea105ceacc5dfc73`, reference=16029.69268099998
- `qnet1_o`: rows=456, cols=1541, integers=1417, nnz=4214, bytes=184240, sha256=`abf1a3597aad6a3ab7d1d0b3343a606490cf49d4a2efa553a2ce795c4898f8c0`, reference=16029.692681
- `r50x360`: rows=410, cols=720, integers=360, nnz=1440, bytes=97596, sha256=`37c38463d072888d4865b63d434b4355812002fae123247d92a470c762e23563`, reference=1653.0
- `roll3000`: rows=2295, cols=1166, integers=738, nnz=29386, bytes=1035164, sha256=`b28b51a69945d6a4c16366582bab87638b6c66f81eb4efef817aba76f6425117`, reference=12889.999992
- `rout`: rows=291, cols=556, integers=315, nnz=2431, bytes=94010, sha256=`ed73ef89a0faffea96a1ef085eb1f64c1116517abd6daaa1b99d3f4cd91eee95`, reference=1077.559999999999
- `sp150x300d`: rows=450, cols=600, integers=300, nnz=1200, bytes=75788, sha256=`339aa4411ef126df72dc66718090fc6915debdc44d732a0f0be903310e52cf4a`, reference=69.0

## Unsolved or unverified results

- `10teams` / sovereign: status=TIME_LIMIT, incumbent_verified=False, best_bound_verified=True, gap=0.0, nodes=5481, class=time_limit, time_to_gap={'0.1': None, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}
- `b-ball` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.21212121212121207, nodes=739803, class=weak_bound, time_to_gap={'0.1': None, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}
- `beasleyC3` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.7973924812860811, nodes=11639, class=weak_bound, time_to_gap={'0.1': None, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}
- `enlight_hard` / sovereign: status=TIME_LIMIT, incumbent_verified=False, best_bound_verified=True, gap=0.0, nodes=156232, class=time_limit, time_to_gap={'0.1': None, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}
- `exp-1-500-5-5` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.7300729210211423, nodes=35846, class=weak_bound, time_to_gap={'0.1': 148.278, '0.01': 427.244, '0.001': None, '0.0001': None, '1e-06': None}
- `markshare_4_0` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=1.0, nodes=253550, class=weak_bound, time_to_gap={'0.1': 44.3293, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}
- `mas76` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.00459579528217333, nodes=308022, class=weak_bound, time_to_gap={'0.1': 1.3913, '0.01': 1.3913, '0.001': 19.6177, '0.0001': 20.212, '1e-06': None}
- `mik-250-20-75-4` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.03768648446601451, nodes=45132, class=weak_bound, time_to_gap={'0.1': 0.890042, '0.01': 35.8387, '0.001': 172.151, '0.0001': None, '1e-06': None}
- `neos-911970` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.32703018322239125, nodes=35189, class=weak_bound, time_to_gap={'0.1': 24.04, '0.01': 105.606, '0.001': None, '0.0001': None, '1e-06': None}
- `neos17` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.03329377622377627, nodes=87789, class=weak_bound, time_to_gap={'0.1': 39.3407, '0.01': 215.572, '0.001': None, '0.0001': None, '1e-06': None}
- `neos5` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.041666666666666664, nodes=406401, class=weak_bound, time_to_gap={'0.1': 2.12577, '0.01': 17.3099, '0.001': None, '0.0001': None, '1e-06': None}
- `noswot` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.04878048780487805, nodes=263777, class=weak_bound, time_to_gap={'0.1': 2.99886, '0.01': 2.99886, '0.001': 2.99886, '0.0001': None, '1e-06': None}
- `p200x1188c` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.5058684880813861, nodes=21763, class=weak_bound, time_to_gap={'0.1': None, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}
- `pg` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.442261431384325, nodes=31699, class=weak_bound, time_to_gap={'0.1': 105.82, '0.01': 115.1, '0.001': None, '0.0001': None, '1e-06': None}
- `qiu` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.004993372103560946, nodes=15189, class=weak_bound, time_to_gap={'0.1': 406.049, '0.01': 594.488, '0.001': None, '0.0001': None, '1e-06': None}
- `qnet1_o` / sovereign: status=OPTIMAL, incumbent_verified=True, best_bound_verified=False, gap=0.0, nodes=555, class=wrong_prune, time_to_gap={'0.1': 169.112, '0.01': 169.112, '0.001': 169.112, '0.0001': 169.112, '1e-06': 169.112}
- `r50x360` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.5329033998920669, nodes=75379, class=weak_bound, time_to_gap={'0.1': 488.708, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}
- `roll3000` / sovereign: status=TIME_LIMIT, incumbent_verified=False, best_bound_verified=True, gap=0.0, nodes=4170, class=time_limit, time_to_gap={'0.1': None, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}
- `rout` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.04245615337329904, nodes=46074, class=weak_bound, time_to_gap={'0.1': 64.8315, '0.01': 336.788, '0.001': 586.949, '0.0001': None, '1e-06': None}
- `sp150x300d` / sovereign: status=FEASIBLE, incumbent_verified=True, best_bound_verified=True, gap=0.3780997333409383, nodes=101301, class=weak_bound, time_to_gap={'0.1': None, '0.01': None, '0.001': None, '0.0001': None, '1e-06': None}

## Branch-and-bound status behavior and correctness risk

- A node LP returning `NUMERICAL_ERROR` or `ITERATION_LIMIT` is retried through the configured LP fallbacks. If it still fails, the subtree is dropped, a warning is recorded, and the final result is downgraded to `FEASIBLE` when an incumbent exists or `NUMERICAL_ERROR` otherwise; it is not treated as infeasible.
- A node LP returning `UNBOUNDED` is propagated as overall MILP `UNBOUNDED`. The finnis temporary-bound path currently returns LP `NUMERICAL_ERROR`, but any future temporary-bound path that returns `UNBOUNDED` would be a correctness risk because the node is not independently proven unbounded.

## Missing input instructions

If this report has missing instances, fetch them with:

```powershell
python benchmarks/runners/run_miplib_stage2.py --fetch
```

The runner uses the official URL recorded in the manifest and never invents replacement data.
