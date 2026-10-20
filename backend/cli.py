import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

from . import store  # noqa: E402  (after logging so the device line prints)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m backend.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("image", type=Path)
    pr.add_argument("--model", action="append", required=True)
    pr.add_argument("--force", action="store_true")
    args = p.parse_args()

    data = args.image.read_bytes()
    for model in args.model:
        rid, meta, cached = store.predict_cached(data, model, force=args.force, source=args.image.name)
        print(f"{'cached' if cached else 'new'}: {store.RUNS / rid}")


if __name__ == "__main__":
    main()
