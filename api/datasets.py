"""Small, explicitly sourced catalogue for reproducible SIH runs."""
import hashlib
from pathlib import Path

from api.routing import summarize

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("afiro", "Netlib LP", "benchmarks/datasets/netlib/afiro.mps", "https://netlib.org/lp/data/"),
    *[(n, "MIPLIB official", f"benchmarks/datasets/miplib/official/{n}.mps", f"https://miplib.zib.de/instance_details_{n}.html")
      for n in ("flugpl", "gt2", "b-ball", "pk1", "gen-ip016")],
    *[(n, "Synthetic robustness", f"benchmarks/datasets/robustness/{n}.json", "Repository fixture")
      for n in ("kuhn_degeneracy", "illconditioned", "weak_lp_relaxation")],
    *[(n, "Synthetic scale", f"benchmarks/datasets/scale/{n}.json", "Repository generator")
      for n in ("transport_20x20", "transport_50x50", "transport_100x100",
                "transport_150x150", "transport_200x200")],
    *[(n, "Industrial example", f"examples/models/{n}.json", "Repository example; not a published industrial dataset")
      for n in ("industrial_refinery_lp", "industrial_blending_lp", "industrial_power_dispatch_lp", "industrial_logistics_milp",
                "industrial_product_mix_lp")],
    *[(n, "QP example", f"examples/models/{n}.json", "Repository example; not QPLIB") for n in ("sample_qp", "qp_ge")],
    *[(n, "GPU showcase", f"benchmarks/datasets/gpu/{n}.json", "Repository generator (benchmarks/tools/generate_gpu_showcase.py)")
      for n in ("plan_600x900", "plan_1000x1500", "plan_1500x2200", "portfolio_qp_600")],
    *[(n, "Sparse scale", f"benchmarks/datasets/sparse/{n}.json", "Repository generator (benchmarks/tools/generate_sparse_scale.py)")
      for n in ("staircase_20x100", "staircase_50x200", "portfolio_qp_5000", "portfolio_qp_20000")],
]


def dataset(dataset_id):
    for name, suite, relative, source in CASES:
        if name == dataset_id:
            path = ROOT / relative
            if not path.is_file():
                raise KeyError(name)
            text = path.read_text(encoding="utf-8")
            fmt = path.suffix.lstrip(".")
            return {"id": name, "suite": suite, "source": source, "modelFormat": fmt,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "shape": summarize(text, fmt), "modelJson": text}
    raise KeyError(dataset_id)


def catalogue():
    items = []
    for name, _, _, _ in CASES:
        try:
            item = dataset(name)
            item.pop("modelJson")
            items.append(item)
        except (KeyError, OSError):
            continue
    return items
