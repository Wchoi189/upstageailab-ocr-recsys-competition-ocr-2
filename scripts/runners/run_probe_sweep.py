#!/usr/bin/env python3
"""Run A/B continuation probes and summarize results from local WandB artifacts.

Usage examples:

  # Print commands only (dry-run)
  uv run python scripts/runners/run_probe_sweep.py

  # Execute both probes and write a ranked summary
  uv run python scripts/runners/run_probe_sweep.py --run

  # Execute with custom checkpoint and target epoch
  uv run python scripts/runners/run_probe_sweep.py \
    --run \
    --checkpoint outputs/checkpoints/best-acc-0.8620.ckpt \
    --target-epochs 50
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.performance.generate_baseline_report import parse_recognition_metrics_from_log


DEFAULT_CHECKPOINT = "outputs/checkpoints/best-acc-0.8620.ckpt"
DEFAULT_EXPERIMENT = "parseq_flash_plateau"
DEFAULT_OUTPUT_ROOT = Path("outputs/reports/probe_sweep")
WANDB_RUN_DIR = Path("outputs/wandb")


@dataclass
class ProbeConfig:
    """Probe configuration payload."""

    name: str
    lr: float
    weight_decay: float
    scheduler_factor: float = 0.7
    scheduler_patience: int = 4


@dataclass
class ProbeResult:
    """Collected probe outcome and extracted metrics."""

    probe_name: str
    command: str
    return_code: int | None
    run_id: str | None
    run_dir: Path | None
    final_val_acc: float | None
    final_val_cer: float | None
    best_val_acc: float | None
    best_val_cer: float | None
    final_val_loss: float | None


def _to_float_or_none(value: Any) -> float | None:
    """Safely convert value to float."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_probe_command(
    checkpoint: str,
    experiment: str,
    target_epochs: int,
    limit_val_batches: int | None,
    trim_pad_to_batch_max: bool,
    probe: ProbeConfig,
) -> list[str]:
    """Build Hydra command arguments for one probe."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/runners/train.py",
        "mode=train",
        f"experiment={experiment}",
        f"checkpoint_path={checkpoint}",
        f"trainer.max_epochs={target_epochs}",
        "trainer.val_check_interval=1.0",
        f"data.sequence.trim_pad_to_batch_max={str(trim_pad_to_batch_max).lower()}",
        f"train.optimizer.lr={probe.lr}",
        f"train.optimizer.weight_decay={probe.weight_decay}",
        f"train.lr_scheduler.factor={probe.scheduler_factor}",
        f"train.lr_scheduler.patience={probe.scheduler_patience}",
        "train.lr_scheduler.monitor=val/acc",
        "train.lr_scheduler.mode=max",
    ]

    if limit_val_batches is not None:
        cmd.append(f"trainer.limit_val_batches={limit_val_batches}")

    return cmd


def command_to_shell_line(command: list[str]) -> str:
    """Render command as one shell-safe line for printing/logging."""
    return " ".join(command)


def find_wandb_run_dirs(root_dir: Path) -> list[Path]:
    """List run-* directories sorted by mtime desc."""
    if not root_dir.exists():
        return []
    run_dirs = [path for path in root_dir.glob("run-*") if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs


def parse_run_id_from_text(output_text: str) -> str | None:
    """Extract wandb run id from training stdout/stderr."""
    patterns = [
        r"/runs/([a-zA-Z0-9]+)",
        r"run-[0-9_]+-([a-zA-Z0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            return match.group(1)
    return None


def find_run_dir_by_id(root_dir: Path, run_id: str) -> Path | None:
    """Find local run directory for a wandb run id."""
    matches = sorted(root_dir.glob(f"run-*-{run_id}"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def find_newest_run_after(root_dir: Path, started_at: datetime) -> Path | None:
    """Fallback: find newest run directory created after probe start."""
    for run_dir in find_wandb_run_dirs(root_dir):
        modified_at = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        if modified_at >= started_at:
            return run_dir
    return None


def extract_metrics_from_run_dir(run_dir: Path) -> dict[str, float | None]:
    """Extract final/best metrics from local run files."""
    summary_file = run_dir / "files" / "wandb-summary.json"
    output_log_file = run_dir / "files" / "output.log"

    final_val_acc: float | None = None
    final_val_cer: float | None = None
    final_val_loss: float | None = None
    best_val_acc: float | None = None
    best_val_cer: float | None = None

    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
            final_val_acc = _to_float_or_none(summary.get("val/acc"))
            final_val_cer = _to_float_or_none(summary.get("val/cer"))
            final_val_loss = _to_float_or_none(summary.get("val_loss"))
        except (json.JSONDecodeError, OSError):
            pass

    if output_log_file.exists():
        parsed = parse_recognition_metrics_from_log(output_log_file)
        points = parsed.get("points") or []
        if points:
            acc_values = [point.get("val_acc") for point in points if point.get("val_acc") is not None]
            cer_values = [point.get("val_cer") for point in points if point.get("val_cer") is not None]
            if acc_values:
                best_val_acc = max(float(acc) for acc in acc_values)
            if cer_values:
                best_val_cer = min(float(cer) for cer in cer_values)

    if best_val_acc is None:
        best_val_acc = final_val_acc
    if best_val_cer is None:
        best_val_cer = final_val_cer

    return {
        "final_val_acc": final_val_acc,
        "final_val_cer": final_val_cer,
        "best_val_acc": best_val_acc,
        "best_val_cer": best_val_cer,
        "final_val_loss": final_val_loss,
    }


def rank_probe_results(results: list[ProbeResult]) -> list[ProbeResult]:
    """Sort by best val/acc desc then best val/cer asc."""

    def sort_key(result: ProbeResult) -> tuple[float, float]:
        acc_value = result.best_val_acc if result.best_val_acc is not None else -1.0
        cer_value = result.best_val_cer if result.best_val_cer is not None else 999.0
        return (-acc_value, cer_value)

    return sorted(results, key=sort_key)


def write_summary(output_dir: Path, ordered_results: list[ProbeResult]) -> None:
    """Write JSON and Markdown summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_payload = []
    for result in ordered_results:
        json_payload.append(
            {
                "probe_name": result.probe_name,
                "command": result.command,
                "return_code": result.return_code,
                "run_id": result.run_id,
                "run_dir": str(result.run_dir) if result.run_dir else None,
                "final_val_acc": result.final_val_acc,
                "final_val_cer": result.final_val_cer,
                "best_val_acc": result.best_val_acc,
                "best_val_cer": result.best_val_cer,
                "final_val_loss": result.final_val_loss,
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(json_payload, indent=2))

    lines = [
        "# Probe Sweep Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Rank | Probe | best val/acc | best val/cer | final val/acc | final val/cer | run_id |",
        "|------|-------|--------------|--------------|---------------|---------------|--------|",
    ]

    for rank, result in enumerate(ordered_results, start=1):
        lines.append(
            "| "
            f"{rank} | {result.probe_name} | "
            f"{result.best_val_acc if result.best_val_acc is not None else 'n/a'} | "
            f"{result.best_val_cer if result.best_val_cer is not None else 'n/a'} | "
            f"{result.final_val_acc if result.final_val_acc is not None else 'n/a'} | "
            f"{result.final_val_cer if result.final_val_cer is not None else 'n/a'} | "
            f"{result.run_id or 'n/a'} |"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="Run A/B continuation probes and summarize metrics")
    parser.add_argument("--run", action="store_true", help="Execute probe commands (default: print only)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint path to resume from")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="Experiment config name")
    parser.add_argument("--target-epochs", type=int, default=50, help="Absolute trainer.max_epochs for continuation")
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=400,
        help="Validation batch cap for probes (use -1 for full validation)",
    )
    parser.add_argument(
        "--trim-pad-to-batch-max",
        action="store_true",
        default=True,
        help="Enable data.sequence.trim_pad_to_batch_max=true (default: true)",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where probe logs and summaries are written",
    )
    return parser.parse_args()


def main() -> int:
    """Entrypoint."""
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    limit_val_batches = None if args.limit_val_batches < 0 else args.limit_val_batches

    probes = [
        ProbeConfig(name="probe_a", lr=5e-5, weight_decay=5e-4),
        ProbeConfig(name="probe_b", lr=8e-5, weight_decay=3e-4),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ProbeResult] = []

    for probe in probes:
        command = build_probe_command(
            checkpoint=str(checkpoint_path),
            experiment=args.experiment,
            target_epochs=args.target_epochs,
            limit_val_batches=limit_val_batches,
            trim_pad_to_batch_max=args.trim_pad_to_batch_max,
            probe=probe,
        )
        shell_line = command_to_shell_line(command)

        print("=" * 90)
        print(f"[{probe.name}] command")
        print(shell_line)

        run_id: str | None = None
        run_dir: Path | None = None
        return_code: int | None = None

        if args.run:
            started_at = datetime.now(tz=timezone.utc)
            completed = subprocess.run(command, text=True, capture_output=True)
            return_code = completed.returncode

            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            combined_text = f"{stdout_text}\n{stderr_text}"

            (output_dir / f"{probe.name}.stdout.log").write_text(stdout_text)
            (output_dir / f"{probe.name}.stderr.log").write_text(stderr_text)

            run_id = parse_run_id_from_text(combined_text)
            if run_id:
                run_dir = find_run_dir_by_id(WANDB_RUN_DIR, run_id)

            if run_dir is None:
                run_dir = find_newest_run_after(WANDB_RUN_DIR, started_at)

            print(f"[{probe.name}] return_code={return_code} run_id={run_id or 'n/a'}")
            if run_dir:
                print(f"[{probe.name}] run_dir={run_dir}")
        else:
            print(f"[{probe.name}] dry-run only (add --run to execute)")

        final_val_acc = None
        final_val_cer = None
        best_val_acc = None
        best_val_cer = None
        final_val_loss = None

        if run_dir and run_dir.exists():
            extracted = extract_metrics_from_run_dir(run_dir)
            final_val_acc = extracted["final_val_acc"]
            final_val_cer = extracted["final_val_cer"]
            best_val_acc = extracted["best_val_acc"]
            best_val_cer = extracted["best_val_cer"]
            final_val_loss = extracted["final_val_loss"]

        all_results.append(
            ProbeResult(
                probe_name=probe.name,
                command=shell_line,
                return_code=return_code,
                run_id=run_id,
                run_dir=run_dir,
                final_val_acc=final_val_acc,
                final_val_cer=final_val_cer,
                best_val_acc=best_val_acc,
                best_val_cer=best_val_cer,
                final_val_loss=final_val_loss,
            )
        )

    ordered_results = rank_probe_results(all_results)
    write_summary(output_dir, ordered_results)

    print("\n" + "=" * 90)
    print("Probe ranking (best val/acc desc, then best val/cer asc):")
    for rank, result in enumerate(ordered_results, start=1):
        print(
            f"{rank}. {result.probe_name} "
            f"best_acc={result.best_val_acc if result.best_val_acc is not None else 'n/a'} "
            f"best_cer={result.best_val_cer if result.best_val_cer is not None else 'n/a'} "
            f"run_id={result.run_id or 'n/a'}"
        )

    print(f"\nSummary files: {output_dir / 'summary.md'}, {output_dir / 'summary.json'}")
    if not args.run:
        print("Dry-run complete. Re-run with --run to execute probes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
