"""Select the approved Stable-only grid without changing historical fit code.

The original runner stays byte-for-byte unchanged so that its validated partial
Stable checkpoint retains its source/config/data signature and can resume.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import fig1e_ed4_6seed_runner as runner


def configure_stable_only() -> None:
    plan = json.loads(runner.PLAN_PATH.read_text(encoding="utf-8"))
    policy = plan.get("selection_policy", {})
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (policy.get("module") != "user_soft.fig1e_ed4_stable_6seed_runner"
            or policy.get("sha256") != digest):
        raise RuntimeError("Stable-only selection policy is not sealed in the campaign plan")
    runner.VERSIONS = ("soft_v1_stable",)
    runner.PLANNED_FITS = 432


def main(argv=None) -> int:
    configure_stable_only()
    return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
