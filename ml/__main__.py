"""Train, evaluate and export a project: ``python -m ml <project>`` (or ``all``)."""

from __future__ import annotations

import argparse
import importlib
import time

PROJECTS = ["rain", "fraud", "insurance", "clustering"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ml", description=__doc__)
    parser.add_argument("project", choices=PROJECTS + ["all"])
    args = parser.parse_args()
    for name in PROJECTS if args.project == "all" else [args.project]:
        module = importlib.import_module(f"ml.{name}")
        start = time.perf_counter()
        meta = module.run()
        print(f"\n== {name} ({time.perf_counter() - start:.0f}s)")
        print(module.summary(meta))


if __name__ == "__main__":
    main()
