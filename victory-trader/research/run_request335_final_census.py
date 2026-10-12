"""Run the registered private census; pause for externally verified date checkpoints."""
import argparse
import json
import os
import time
from pathlib import Path

from victory_trader.offline_source_inventory import canonical, strict_json
from victory_trader.r335_alpaca_qualification import load_plan
from victory_trader.r335_final_census import collect
from victory_trader.r335_atomic_checkpoint import make_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--prefix-proofs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    args = parser.parse_args()
    if args.credentials.stat().st_mode & 0o077 or args.credentials.stat().st_size > 16384:
        raise ValueError("small_private_credentials_required")
    credentials = strict_json(args.credentials.read_bytes())
    secrets = (credentials.get("APCA_API_KEY_ID"), credentials.get("APCA_API_SECRET_KEY"))
    plan = load_plan(Path("research/request335-free-source-plan.json.gz"))
    args.control.mkdir(parents=True, mode=0o700, exist_ok=True)
    def persist(root, day, manifest):
        archive = args.control / f"MoneyMaker-R335-private-census-20261012-{day}.tar.gz"
        value = make_archive(root, archive, day=day, secrets=secrets)
        proof = {"day": day, **value, "saved": False, "retrieved_archive_hash_equal": False}
        pending = {"day": day, "archive": str(archive), "proof": proof,
                   "http_attempts": manifest["http_attempts"]}
        path = args.control / f"{day}.pending.json"
        path.write_bytes(canonical(pending))
        path.chmod(0o600)
        print("PRIVATE_CHECKPOINT_READY " + json.dumps({"day": day, **value}), flush=True)
        verified = args.control / f"{day}.verified.json"
        started = time.monotonic()
        while time.monotonic() - started < 600:
            if verified.is_file():
                ack = strict_json(verified.read_bytes())
                if any(ack.get(key) != proof[key] for key in ("day", "archive_sha256", "archive_bytes", "private_files")):
                    raise ValueError("private_checkpoint_confirmation_changed")
                return ack
            time.sleep(0.5)
        raise ValueError("private_checkpoint_confirmation_timeout")
    os.umask(0o077)
    result = collect(plan, args.prefix, args.evidence, credentials=secrets,
                     prefix_proofs=strict_json(args.prefix_proofs.read_bytes()),
                     attestation=args.attestation.read_bytes(), persist=persist)
    print("COLLECTION_FINISHED " + json.dumps({"http_attempts": result["http_attempts"],
          "stop_reason": result["stop_reason"], "verified_date_checkpoints": len(result["private_checkpoints"])}), flush=True)


if __name__ == "__main__":
    main()
