"""Problem E stages. Science stages (eda, classify, allocate, forecast, results, figures, tables) go here.

Conventions (see docs/ENGINEERING_STANDARD.md):
  * every stage is ``@stage(name, deps=(...))`` and returns a JSON-serialisable metrics dict;
  * outputs go only to ``ctx.out(...)``; paper numbers via ``ctx.number("Key", value)``;
  * ``results`` writes result*.xlsx with ``forge.xlsx.write_result`` from the organisers' templates;
  * every forecast carries a backtest and every allocation an out-of-sample comparison.
"""

from __future__ import annotations

from typing import Any

from forge.context import StageContext
from forge.runner import stage
from pipelines.common.validation import run_validation
from pipelines.e.contracts import INPUT_CONTRACTS


@stage("validate", deps=("ingest",), description="Validate the three SEM sheets against their contracts")
def validate(ctx: StageContext) -> dict[str, Any]:
    return run_validation(ctx, INPUT_CONTRACTS)
