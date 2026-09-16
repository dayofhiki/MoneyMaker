from __future__ import annotations

import argparse
import json

from .config import load_settings
from .massive_client import MassiveClient


def check_api() -> int:
    settings = load_settings()
    client = MassiveClient(settings.massive_api_key)
    payload = client.previous_close("AAPL")
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="victory-trader")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-api", help="Check Massive API connectivity using AAPL previous close")
    args = parser.parse_args()

    if args.command == "check-api":
        return check_api()
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
