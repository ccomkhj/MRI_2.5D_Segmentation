"""Compatibility wrapper for the modular training CLI."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri.cli.train import main as train_main


def main(argv=None) -> int:
    warnings.warn(
        "service/train.py is a compatibility wrapper. Prefer mri/cli/train.py or scripts/new/train.",
        DeprecationWarning,
        stacklevel=2,
    )
    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
