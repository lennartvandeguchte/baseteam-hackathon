"""CLI: `supply-risk run --supplier ... --product ...` and `supply-risk smoke`."""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="supply-risk", description="Autonomous supply chain risk analysis agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Analyse one supplier and write a report")
    run.add_argument("--supplier", required=True)
    run.add_argument("--product", required=True)
    run.add_argument("--buyer")
    run.add_argument("--single-source", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--used-in")
    run.add_argument("--country", help="Supplier country hint")

    smoke = sub.add_parser("smoke", help="Tool-calling smoke test of candidate Nebius models")
    smoke.add_argument("--models", nargs="*")

    args = parser.parse_args(argv)
    if args.command == "smoke":
        from supply_risk.smoke import CANDIDATES, run_smoke

        for line in run_smoke(args.models or CANDIDATES):
            print(line, flush=True)
        return

    from supply_risk.agent import Request, run as run_agent

    request = Request(
        supplier=args.supplier, product=args.product, buyer=args.buyer,
        single_source=args.single_source, used_in=args.used_in, country=args.country,
    )
    slug = re.sub(r"[^a-z0-9]+", "-", args.supplier.lower()).strip("-")
    run_dir = Path("runs") / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}"
    start = time.monotonic()
    report = run_agent(request, run_dir, on_step=lambda step: print(f"[{time.monotonic() - start:6.1f}s] {step}", flush=True))
    print(f"\nReport: {report if report.exists() else 'NOT WRITTEN'} ({time.monotonic() - start:.0f}s)")


if __name__ == "__main__":
    main()
