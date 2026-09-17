#!/usr/bin/env python3
"""Askable arm: Jev chains a fixed primitive catalog. No PPO.

Runs tasks in phase order (motion → reach → manipulate → buttons).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_robotics.common import JEVClient, jsonable, load_local_env
from jev_robotics.primitives import KINDS, TASKS, PrimitiveChooser, run_task


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="1,2,3,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--only", default="")
    parser.add_argument(
        "--out", default="outputs/jev_robotics/ask_arm/results.json"
    )
    args = parser.parse_args()

    phases = {int(p) for p in args.phases.split(",") if p.strip()}
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    tasks = [
        task
        for task in TASKS()
        if task.phase in phases and (not wanted or task.name in wanted)
    ]
    print(
        json.dumps(
            {
                "primitive_count": len(KINDS),
                "primitives": list(KINDS),
                "tasks": [task.name for task in tasks],
            }
        ),
        flush=True,
    )
    chooser = PrimitiveChooser(
        min_confidence=args.min_confidence, client=JEVClient()
    )
    results = []
    for task in tasks:
        result = run_task(task, seed=args.seed, chooser=chooser)
        results.append(result)
        print(json.dumps(jsonable(result), sort_keys=True), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(r["success"] for r in results)
    summary = {
        "passed": passed,
        "total": len(results),
        "primitive_count": len(KINDS),
        "results": results,
    }
    out.write_text(json.dumps(jsonable(summary), indent=2))
    print(
        f"ask-arm {passed}/{len(results)}  primitives={len(KINDS)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
