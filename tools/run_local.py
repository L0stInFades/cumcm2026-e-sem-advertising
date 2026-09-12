#!/usr/bin/env python3
"""Run the Problem-E pipeline on this machine, without Modal.

The reference execution of this repository is in the cloud (``tools/cli.py``), but an evaluator has no
account there. This script is the local equivalent: it calls exactly the same stage functions, in the
same order, through the same runner, writing into a local directory instead of a cloud volume.

    python3 tools/run_local.py                       # ingest -> ... -> tables, into ./local_runs/<id>
    python3 tools/run_local.py --stages ingest,validate --out /tmp/e
    python3 tools/run_local.py --quick               # reduced budgets: minutes rather than hours

Requirements (pip install): pandas, numpy, scipy, scikit-learn, statsmodels, openpyxl, pyarrow, matplotlib, cvxpy.
The organisers' attachments must be present in ``附件/`` exactly as delivered; nothing else is needed.
Typesetting stages (``paper``/``qa``/``package``/``release``) need a TeX Live installation with XeLaTeX
and the CJK fonts, so they are not part of the default list here — the scientific stages are.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STAGES = "ingest,validate,eda,classify,allocate,forecast,results,figures,tables"
# --quick reduces the Monte-Carlo scenario count and the bootstrap draws through ``config_overrides``
# (the same mechanism as ``--profile``): minutes instead of tens of minutes. Point estimates are
# unchanged; interval widths and bootstrap intervals are noisier than the published ones.
_QUICK_CONFIG = {"allocate": {"bootstrap": 60}, "forecast": {"scenarios": 100}}
QUICK = {
    s: {"config_overrides": _QUICK_CONFIG}
    for s in ("eda", "classify", "allocate", "forecast", "results", "figures", "tables")
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=DEFAULT_STAGES, help=f"comma-separated (default: {DEFAULT_STAGES})")
    ap.add_argument("--out", default=str(REPO / "local_runs"), help="directory that plays the role of the volume")
    ap.add_argument("--run-id", default=None, help="reuse an existing run id instead of creating one")
    ap.add_argument("--quick", action="store_true", help="small solver budgets (minutes instead of hours)")
    ap.add_argument("--force", action="store_true", help="re-run stages that already completed")
    ap.add_argument("--param", action="append", default=[], help="stage param, e.g. workers=4 or all:workers=4")
    ns = ap.parse_args()

    sys.path.insert(0, str(REPO))
    import pipelines.e.stages  # noqa: F401 - registers the stages
    from forge.runner import execute

    vol = Path(ns.out).resolve()
    run_id = ns.run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S-local")
    (vol / "runs" / run_id).mkdir(parents=True, exist_ok=True)
    extra: dict[str, dict[str, object]] = {}
    for item in ns.param:
        scope, _, rest = item.partition(":") if ":" in item else ("all", "", item)
        key, _, value = rest.partition("=")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        extra.setdefault(scope, {})[key] = parsed

    print(f"repo={REPO}\nvolume={vol}\nrun_id={run_id}")
    for stage in [s.strip() for s in ns.stages.split(",") if s.strip()]:
        params: dict[str, object] = {"size": "local"}
        if ns.quick:
            params.update(QUICK.get(stage, {}))
        params.update(extra.get("all", {}))
        params.update(extra.get(stage, {}))
        t0 = time.monotonic()
        result = execute(stage_name=stage, run_id=run_id, params=params, force=ns.force, repo=REPO, vol=vol)
        status = result.get("status")
        print(f"[{time.monotonic() - t0:8.1f}s] {stage:18s} {status}")
        if status == "failed":
            print(json.dumps(result.get("error", {}), ensure_ascii=False, indent=2))
            return 1
    print(f"\nresults: {vol / 'runs' / run_id / 'results'}  (result1-4.xlsx)")
    print(f"figures: {vol / 'runs' / run_id / 'figures' / 'figures'}")
    print(f"tables:  {vol / 'runs' / run_id / 'tables' / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
