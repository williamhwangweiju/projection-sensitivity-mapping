#!/usr/bin/env python3
"""Copy this hybrid implementation over an existing repository without deleting files."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", "data"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    overlay = args.overlay.resolve()
    target = args.target.resolve()
    if not (overlay / "src" / "mapping" / "digital_selection.py").is_file():
        raise FileNotFoundError("Overlay does not look like the hybrid package.")
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in overlay.rglob("*"):
        relative = source.relative_to(overlay)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        destination = target / relative
        if source.is_dir():
            if not args.dry_run:
                destination.mkdir(parents=True, exist_ok=True)
            continue
        print(f"{source} -> {destination}")
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        copied += 1
    print(f"{'Would copy' if args.dry_run else 'Copied'} {copied} files.")


if __name__ == "__main__":
    main()
