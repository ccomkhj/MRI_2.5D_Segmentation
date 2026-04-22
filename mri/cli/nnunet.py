"""Standalone MONAI nnU-Net V2 runner entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mri.nnunet.export import export_nnunet_workspace, parse_split_args, update_manifest_stage


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Segmentation task config used to export nnU-Net samples")
    parser.add_argument("--run-name", help="nnU-Net workspace name. Defaults to nnunet-<task-config-stem>")
    parser.add_argument("--output-root", default="experiments/nnunet", help="Root directory for nnU-Net workspaces")
    parser.add_argument(
        "--training-splits",
        default="train,val",
        help="Comma-separated split keys exported into the nnU-Net training pool",
    )
    parser.add_argument(
        "--testing-splits",
        default="test",
        help="Comma-separated split keys exported into the nnU-Net testing pool",
    )
    parser.add_argument("--dataset-name-or-id", type=int, default=901, help="Dataset id passed to nnUNetV2Runner")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing nnU-Net workspace")
    parser.add_argument("--dry-run", action="store_true", help="Prepare/export without launching MONAI nnU-Net")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and launch MONAI nnUNetV2Runner workflows")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export sample-level nnU-Net workspace")
    _add_common_args(export_parser)

    convert_parser = subparsers.add_parser("convert_dataset", help="Export and run nnUNetV2Runner convert_dataset")
    _add_common_args(convert_parser)

    plan_parser = subparsers.add_parser("plan_and_process", help="Export and run nnUNetV2Runner plan_and_process")
    _add_common_args(plan_parser)

    train_parser = subparsers.add_parser("train_single_model", help="Export and run nnUNetV2Runner train_single_model")
    _add_common_args(train_parser)
    train_parser.add_argument("--nnunet-config", default="2d", help="nnU-Net configuration, e.g. 2d or 3d_fullres")
    train_parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold passed to nnUNetV2Runner")
    train_parser.add_argument("--gpu-id", default="0", help="GPU id or comma-separated GPU ids")
    train_parser.add_argument(
        "--trainer-class-name",
        default="nnUNetTrainer_1epoch",
        help="nnU-Net trainer class, e.g. nnUNetTrainer or nnUNetTrainer_1epoch",
    )

    run_parser = subparsers.add_parser("run", help="Export and run the full nnUNetV2Runner pipeline")
    _add_common_args(run_parser)
    run_parser.add_argument(
        "--trainer-class-name",
        default="nnUNetTrainer_1epoch",
        help="nnU-Net trainer class used by nnUNetV2Runner run",
    )

    return parser


def _default_run_name(config_path: str) -> str:
    return f"nnunet-{Path(config_path).stem}"


def _runner_command(args, input_yaml: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "monai.apps.nnunet",
        "nnUNetV2Runner",
        args.command,
        "--input_config",
        input_yaml,
    ]

    if args.command == "train_single_model":
        command.extend(
            [
                "--config",
                args.nnunet_config,
                "--fold",
                str(args.fold),
                "--gpu_id",
                str(args.gpu_id),
                "--trainer_class_name",
                args.trainer_class_name,
            ]
        )
    elif args.command == "run":
        command.extend(["--trainer_class_name", args.trainer_class_name])

    return command


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_name = args.run_name or _default_run_name(args.config)
    training_splits, testing_splits = parse_split_args(args.training_splits, args.testing_splits)
    manifest = export_nnunet_workspace(
        config_path=args.config,
        output_root=args.output_root,
        run_name=run_name,
        training_splits=training_splits,
        testing_splits=testing_splits,
        dataset_name_or_id=args.dataset_name_or_id,
        force=args.force,
        dry_run=args.dry_run,
    )

    if args.command == "export":
        return 0

    manifest_path = Path(manifest["artifacts"]["run_root"]) / "manifests" / "nnunet_manifest.json"
    command = _runner_command(args, manifest["artifacts"]["input_yaml"])
    update_manifest_stage(
        manifest_path,
        stage_name=args.command,
        status="planned" if args.dry_run else "running",
        command=command,
    )

    if args.dry_run:
        return 0

    completed = subprocess.run(command, check=False)
    status = "completed" if completed.returncode == 0 else "failed"
    update_manifest_stage(manifest_path, stage_name=f"{args.command}_result", status=status, command=command)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
