"""Compatibility wrapper for the modular inference CLI."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri.cli.infer import main as infer_main


def main(argv=None) -> int:
    warnings.warn(
        "service/inference.py is a compatibility wrapper. Prefer mri/cli/infer.py or scripts/new/inference.",
        DeprecationWarning,
        stacklevel=2,
    )
    return infer_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
