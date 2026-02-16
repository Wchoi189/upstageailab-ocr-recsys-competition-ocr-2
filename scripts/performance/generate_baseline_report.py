#!/usr/bin/env python3
"""
Generate baseline performance report from WandB profiling run.

This script fetches performance metrics from a WandB run and generates
a comprehensive markdown report documenting current bottlenecks.

Usage:
    uv run python scripts/performance/generate_baseline_report.py \
        --run-id <wandb_run_id> \
        --output docs/reports/baseline_2026-02-16.md \
        --project receipt-text-recognition-ocr-project
"""

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _to_float(value: Any, default: float = 0.0) -> float:
    """Safely convert metric values to float and skip NaN/None."""
    try:
        converted = float(value)
        if math.isnan(converted):
            return default
        return converted
    except (TypeError, ValueError):
        return default


def parse_recognition_metrics_from_log(log_path: Path) -> dict[str, Any]:
    """Parse recognition metrics from local wandb output.log using robust regex."""
    if not log_path.exists():
        return {
            "source": "missing",
            "path": str(log_path),
            "points": [],
            "raw_metric_lines": 0,
            "parse_fail_lines": 0,
            "match_ratios": [],
        }

    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    metric_re = re.compile(r"val/acc:\s*([0-9]+(?:\.[0-9]+)?).*?val/cer:\s*([0-9]+(?:\.[0-9]+)?)")
    epoch_progress_re = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    step_progress_re = re.compile(r"(\d+)\s*/\s*(\d+)")
    match_count_re = re.compile(r"Match Count:\s*(\d+)\s*/\s*(\d+)")

    points: list[dict[str, float]] = []
    raw_metric_lines = 0
    parse_fail_lines = 0
    match_ratios: list[float] = []

    for line in log_path.read_text(errors="ignore").splitlines():
        clean = ansi_re.sub("", line).replace("\r", "")

        if "val/acc:" in clean and "val/cer:" in clean:
            raw_metric_lines += 1
            metric_match = metric_re.search(clean)
            if metric_match:
                points.append(
                    {
                        "val_acc": _to_float(metric_match.group(1), float("nan")),
                        "val_cer": _to_float(metric_match.group(2), float("nan")),
                        "epoch_current": float("nan"),
                        "epoch_total": float("nan"),
                        "step_in_epoch": float("nan"),
                        "steps_per_epoch": float("nan"),
                    }
                )

                epoch_match = epoch_progress_re.search(clean)
                if epoch_match:
                    points[-1]["epoch_current"] = _to_float(epoch_match.group(1), float("nan"))
                    points[-1]["epoch_total"] = _to_float(epoch_match.group(2), float("nan"))

                # Capture the closest step-progress pair near the metric text.
                # Use the last pair to avoid grabbing unrelated small fractions from text noise.
                step_matches = list(step_progress_re.finditer(clean))
                if step_matches:
                    step_match = step_matches[-1]
                    points[-1]["step_in_epoch"] = _to_float(step_match.group(1), float("nan"))
                    points[-1]["steps_per_epoch"] = _to_float(step_match.group(2), float("nan"))
            else:
                parse_fail_lines += 1

        match_count_match = match_count_re.search(clean)
        if match_count_match:
            numer = _to_float(match_count_match.group(1), 0.0)
            denom = _to_float(match_count_match.group(2), 0.0)
            if denom > 0:
                match_ratios.append(numer / denom)

    # Deduplicate consecutive repeated progress rows
    dedup_points: list[dict[str, float]] = []
    for point in points:
        if not dedup_points:
            dedup_points.append(point)
            continue

        prev = dedup_points[-1]
        if abs(prev["val_acc"] - point["val_acc"]) < 1e-12 and abs(prev["val_cer"] - point["val_cer"]) < 1e-12:
            continue
        dedup_points.append(point)

    return {
        "source": "local_output_log_regex",
        "path": str(log_path),
        "points": dedup_points,
        "raw_metric_lines": raw_metric_lines,
        "parse_fail_lines": parse_fail_lines,
        "match_ratios": match_ratios,
    }


def fetch_wandb_metrics(run_id: str, project: str, entity: str | None = None) -> dict[str, Any]:
    """
    Fetch performance metrics from a WandB run.

    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity (optional, uses default if None)

    Returns:
        Dictionary containing run metrics and metadata
    """
    if not WANDB_AVAILABLE:
        raise ImportError("WandB is not installed. Install with: uv add wandb")

    # Initialize wandb API
    api = wandb.Api()

    # Fetch run
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"

    try:
        run = api.run(run_path)
    except Exception as e:
        raise ValueError(f"Failed to fetch run {run_path}: {e}")

    # Extract performance metrics
    metrics = {
        "run_id": run_id,
        "run_name": run.name,
        "created_at": run.created_at,
        "state": run.state,
        "config": run.config,
        "summary": run.summary._json_dict,
        "history": [],
        "local_parse": None,
    }

    # Fetch history (time-series metrics)
    history_df = run.history(
        keys=[
            "performance/val_epoch_time",
            "performance/val_batch_mean",
            "performance/val_batch_median",
            "performance/val_batch_p95",
            "performance/val_batch_p99",
            "performance/val_batch_std",
            "performance/val_num_batches",
            "performance/gpu_memory_gb",
            "performance/gpu_memory_reserved_gb",
            "performance/cpu_memory_percent",
            "train/loss",
            "val_loss",
            "val/acc",
            "val/cer",
            "epoch",
            "trainer/global_step",
            "lr",
            "train/lr",
            "optimizer/lr",
            "lr-Adam",
            "lr/pg0",
        ]
    )

    metrics["history"] = history_df.to_dict("records") if not history_df.empty else []

    # Local fallback parsing: useful when WandB history is sparse/trimmed
    local_log_candidates = sorted(Path("outputs/wandb/wandb").glob(f"run-*-{run_id}/files/output.log"))
    if local_log_candidates:
        metrics["local_parse"] = parse_recognition_metrics_from_log(local_log_candidates[-1])

    return metrics


def analyze_bottlenecks(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze performance metrics to identify bottlenecks.

    Args:
        metrics: Raw metrics from WandB

    Returns:
        Analysis results with bottleneck identification
    """
    summary = metrics.get("summary", {})
    history = metrics.get("history", [])
    local_parse = metrics.get("local_parse") or {}

    # Check if performance profiling metrics are available
    has_performance_metrics = any(key.startswith("performance/") for key in summary.keys())
    has_recognition_metrics = any(key in summary for key in ["val/acc", "val/cer", "val_loss", "train/loss"])

    val_acc_history = [_to_float(item.get("val/acc"), float("nan")) for item in history]
    val_acc_history = [v for v in val_acc_history if not math.isnan(v)]

    val_cer_history = [_to_float(item.get("val/cer"), float("nan")) for item in history]
    val_cer_history = [v for v in val_cer_history if not math.isnan(v)]

    trend_source = "wandb_history"
    parsed_points = local_parse.get("points", []) if isinstance(local_parse.get("points", []), list) else []

    if not val_acc_history and parsed_points:
        trend_source = "local_output_log_regex"
        val_acc_history = [_to_float(point.get("val_acc"), float("nan")) for point in parsed_points]
        val_acc_history = [v for v in val_acc_history if not math.isnan(v)]
        val_cer_history = [_to_float(point.get("val_cer"), float("nan")) for point in parsed_points]
        val_cer_history = [v for v in val_cer_history if not math.isnan(v)]

    best_point: dict[str, Any] | None = None
    final_point: dict[str, Any] | None = None
    if trend_source == "local_output_log_regex" and parsed_points:
        valid_points = [point for point in parsed_points if not math.isnan(_to_float(point.get("val_acc"), float("nan"),))]
        if valid_points:
            best_point = max(valid_points, key=lambda point: _to_float(point.get("val_acc"), 0.0))
            final_point = valid_points[-1]

    val_loss_history = [_to_float(item.get("val_loss"), float("nan")) for item in history]
    val_loss_history = [v for v in val_loss_history if not math.isnan(v)]

    train_loss_history = [_to_float(item.get("train/loss"), float("nan")) for item in history]
    train_loss_history = [v for v in train_loss_history if not math.isnan(v)]

    lr_candidates = ["lr", "train/lr", "optimizer/lr", "lr-Adam", "lr/pg0"]
    lr_history: list[float] = []
    for key in lr_candidates:
        vals = [_to_float(item.get(key), float("nan")) for item in history]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            lr_history = vals
            break

    # Define explicit types for the analysis dictionary
    analysis: dict[str, Any] = {
        "validation_time": {
            "total_seconds": summary.get("performance/val_epoch_time", 0),
            "batch_mean_ms": summary.get("performance/val_batch_mean", 0) * 1000,
            "batch_median_ms": summary.get("performance/val_batch_median", 0) * 1000,
            "batch_p95_ms": summary.get("performance/val_batch_p95", 0) * 1000,
            "batch_p99_ms": summary.get("performance/val_batch_p99", 0) * 1000,
            "batch_std_ms": summary.get("performance/val_batch_std", 0) * 1000,
            "num_batches": summary.get("performance/val_num_batches", 0),
        },
        "memory_usage": {
            "gpu_memory_gb": summary.get("performance/gpu_memory_gb", 0),
            "gpu_memory_reserved_gb": summary.get("performance/gpu_memory_reserved_gb", 0),
            "cpu_memory_percent": summary.get("performance/cpu_memory_percent", 0),
        },
        "training_metrics": {
            "val_hmean": summary.get("val/hmean", 0),
            "val_precision": summary.get("val/precision", 0),
            "val_recall": summary.get("val/recall", 0),
            "test_hmean": summary.get("test/hmean", 0),
            "test_precision": summary.get("test/precision", 0),
            "test_recall": summary.get("test/recall", 0),
            "train_loss": summary.get("train/loss", 0),
            "val_loss": summary.get("val_loss", 0),
            "val_acc": summary.get("val/acc", 0),
            "val_cer": summary.get("val/cer", 0),
            "epoch": summary.get("epoch", 0),
            "global_step": summary.get("trainer/global_step", 0),
        },
        "has_performance_profiling": has_performance_metrics,
        "has_recognition_metrics": has_recognition_metrics,
        "recognition_trend": {
            "source": trend_source,
            "history_points": len(val_acc_history),
            "best_val_acc": max(val_acc_history) if val_acc_history else _to_float(summary.get("val/acc", 0.0)),
            "final_val_acc": val_acc_history[-1] if val_acc_history else _to_float(summary.get("val/acc", 0.0)),
            "best_val_cer": min(val_cer_history) if val_cer_history else _to_float(summary.get("val/cer", 0.0)),
            "final_val_cer": val_cer_history[-1] if val_cer_history else _to_float(summary.get("val/cer", 0.0)),
            "min_val_loss": min(val_loss_history) if val_loss_history else _to_float(summary.get("val_loss", 0.0)),
            "final_val_loss": val_loss_history[-1] if val_loss_history else _to_float(summary.get("val_loss", 0.0)),
            "final_train_loss": train_loss_history[-1] if train_loss_history else _to_float(summary.get("train/loss", 0.0)),
            "lr_first": lr_history[0] if lr_history else 0.0,
            "lr_final": lr_history[-1] if lr_history else 0.0,
            "lr_max": max(lr_history) if lr_history else 0.0,
            "lr_min": min(lr_history) if lr_history else 0.0,
            "regex_raw_metric_lines": _to_float(local_parse.get("raw_metric_lines"), 0.0),
            "regex_parse_fail_lines": _to_float(local_parse.get("parse_fail_lines"), 0.0),
            "regex_match_ratio_best": max(local_parse.get("match_ratios", [0.0])) if local_parse.get("match_ratios") else 0.0,
            "regex_match_ratio_final": local_parse.get("match_ratios", [0.0])[-1] if local_parse.get("match_ratios") else 0.0,
            "best_epoch_hint": _to_float(best_point.get("epoch_current"), float("nan")) if best_point else float("nan"),
            "best_epoch_total_hint": _to_float(best_point.get("epoch_total"), float("nan")) if best_point else float("nan"),
            "best_step_hint": _to_float(best_point.get("step_in_epoch"), float("nan")) if best_point else float("nan"),
            "best_steps_per_epoch_hint": _to_float(best_point.get("steps_per_epoch"), float("nan")) if best_point else float("nan"),
            "final_epoch_hint": _to_float(final_point.get("epoch_current"), float("nan")) if final_point else float("nan"),
            "final_epoch_total_hint": _to_float(final_point.get("epoch_total"), float("nan")) if final_point else float("nan"),
            "final_step_hint": _to_float(final_point.get("step_in_epoch"), float("nan")) if final_point else float("nan"),
            "final_steps_per_epoch_hint": _to_float(final_point.get("steps_per_epoch"), float("nan")) if final_point else float("nan"),
        },
        "bottlenecks": [],
        "comparisons": {},
    }

    if has_recognition_metrics:
        trend = analysis["recognition_trend"]
        acc_drop = trend["best_val_acc"] - trend["final_val_acc"]
        cer_rise = trend["final_val_cer"] - trend["best_val_cer"]

        if acc_drop > 0.01:
            analysis["bottlenecks"].append(
                {
                    "type": "Validation regression after peak",
                    "description": f"Best val/acc ({trend['best_val_acc']:.4f}) fell to final ({trend['final_val_acc']:.4f}), drop={acc_drop:.4f}",
                    "severity": "HIGH" if acc_drop > 0.02 else "MEDIUM",
                }
            )

        if cer_rise > 0.01:
            analysis["bottlenecks"].append(
                {
                    "type": "Character error rate worsened",
                    "description": f"Best val/cer ({trend['best_val_cer']:.4f}) increased to final ({trend['final_val_cer']:.4f}), delta={cer_rise:.4f}",
                    "severity": "MEDIUM",
                }
            )

        min_val_loss = trend["min_val_loss"]
        final_val_loss = trend["final_val_loss"]
        if min_val_loss > 0 and final_val_loss > min_val_loss * 1.1:
            analysis["bottlenecks"].append(
                {
                    "type": "Validation loss drift",
                    "description": f"Final val_loss ({final_val_loss:.4f}) is above best observed ({min_val_loss:.4f})",
                    "severity": "MEDIUM",
                }
            )

        if trend["lr_max"] > 0 and trend["lr_first"] > 0 and trend["lr_max"] >= trend["lr_first"] * 1.5:
            analysis["bottlenecks"].append(
                {
                    "type": "Aggressive LR excursion",
                    "description": f"Observed LR peak ({trend['lr_max']:.2e}) significantly above starting LR ({trend['lr_first']:.2e})",
                    "severity": "MEDIUM",
                }
            )

        # Lightweight phase-level trend summary for educational interpretation
        if len(val_acc_history) >= 6:
            n = len(val_acc_history)
            first_slice = val_acc_history[: max(1, n // 3)]
            mid_slice = val_acc_history[n // 3 : (2 * n) // 3]
            last_slice = val_acc_history[(2 * n) // 3 :]
            analysis["comparisons"]["recognition_phase"] = {
                "early_acc": sum(first_slice) / len(first_slice),
                "mid_acc": sum(mid_slice) / len(mid_slice),
                "late_acc": sum(last_slice) / len(last_slice),
            }

    # Only analyze performance bottlenecks if metrics are available
    if has_performance_metrics:
        # Identify bottlenecks
        batch_mean = analysis["validation_time"]["batch_mean_ms"]
        batch_p95 = analysis["validation_time"]["batch_p95_ms"]
        batch_std = analysis["validation_time"]["batch_std_ms"]

        if batch_p95 > batch_mean * 1.5:
            analysis["bottlenecks"].append(
                {
                    "type": "High variance in batch times",
                    "description": f"P95 ({batch_p95:.1f}ms) is {batch_p95 / batch_mean:.1f}x the mean ({batch_mean:.1f}ms)",
                    "severity": "HIGH",
                }
            )

        if batch_std > batch_mean * 0.5:  # If standard deviation is more than 50% of mean
            analysis["bottlenecks"].append(
                {
                    "type": "High variability in batch processing",
                    "description": f"Standard deviation ({batch_std:.1f}ms) is {batch_std / batch_mean:.1f}x the mean ({batch_mean:.1f}ms)",
                    "severity": "MEDIUM",
                }
            )

        # PyClipper bottleneck check based on performance_optimization_plan.md
        if batch_mean > 1000:  # More than 1 second per batch is considered slow
            analysis["bottlenecks"].append(
                {
                    "type": "Slow validation bottleneck likely due to PyClipper",
                    "description": f"Average batch time ({batch_mean:.1f}ms) is significantly high, likely due to PyClipper polygon processing",
                    "severity": "HIGH",
                }
            )

        # Add training comparison if available from config or other sources
        # This would typically require comparing with training metrics as well
        train_batch_time = metrics.get("config", {}).get("train_batch_time", 0) * 1000
        if train_batch_time > 0:
            slowdown_ratio = batch_mean / train_batch_time if train_batch_time > 0 else 0
            analysis["comparisons"]["validation_slowdown"] = {
                "ratio": slowdown_ratio,
                "description": f"Validation is {slowdown_ratio:.1f}x slower than training",
            }
    else:
        # Analyze training metrics for insights
        if not has_recognition_metrics:
            val_hmean = analysis["training_metrics"]["val_hmean"]
            if val_hmean < 0.8:
                analysis["bottlenecks"].append(
                    {
                        "type": "Low validation performance",
                        "description": f"Validation H-mean ({val_hmean:.3f}) is below 0.8, indicating potential model issues",
                        "severity": "HIGH",
                    }
                )

        train_loss = analysis["training_metrics"]["train_loss"]
        val_loss = analysis["training_metrics"]["val_loss"]
        if val_loss > train_loss * 2:
            analysis["bottlenecks"].append(
                {
                    "type": "Overfitting detected",
                    "description": f"Validation loss ({val_loss:.3f}) is significantly higher than training loss ({train_loss:.3f})",
                    "severity": "MEDIUM",
                }
            )

    return analysis


def generate_markdown_report(
    metrics: dict[str, Any],
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate a markdown report from metrics and analysis.

    Args:
        metrics: Raw metrics from WandB
        analysis: Bottleneck analysis results
        output_path: Path to save the markdown report
    """
    report_lines: list[str] = []

    # Header
    report_lines.extend(
        [
            "# Performance Baseline Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**WandB Run:** [{metrics['run_name']}](https://wandb.ai/runs/{metrics['run_id']})",
            f"**Run ID:** `{metrics['run_id']}`",
            f"**Status:** {metrics['state']}",
            "",
            "---",
            "",
        ]
    )

    # Check if performance profiling was enabled
    has_performance = analysis.get("has_performance_profiling", False)

    if has_performance:
        # Validation Performance Section (only if performance metrics available)
        val_time = analysis["validation_time"]
        report_lines.extend(
            [
                "## Validation Performance",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| **Total Validation Time** | {val_time['total_seconds']:.2f}s |",
                f"| **Number of Batches** | {val_time['num_batches']} |",
                f"| **Mean Batch Time** | {val_time['batch_mean_ms']:.1f}ms |",
                f"| **Median Batch Time** | {val_time['batch_median_ms']:.1f}ms |",
                f"| **P95 Batch Time** | {val_time['batch_p95_ms']:.1f}ms |",
                f"| **P99 Batch Time** | {val_time['batch_p99_ms']:.1f}ms |",
                f"| **Batch Time Std Dev** | {val_time['batch_std_ms']:.1f}ms |",
                "",
            ]
        )

        # Memory Usage
        mem = analysis["memory_usage"]
        report_lines.extend(
            [
                "## Memory Usage",
                "",
                "| Resource | Usage |",
                "|----------|-------|",
                f"| **GPU Memory** | {mem['gpu_memory_gb']:.2f} GB |",
                f"| **GPU Memory Reserved** | {mem['gpu_memory_reserved_gb']:.2f} GB |",
                f"| **CPU Memory** | {mem['cpu_memory_percent']:.1f}% |",
                "",
            ]
        )
    else:
        # Training Metrics Section (when no performance profiling)
        train_metrics = analysis["training_metrics"]
        if analysis.get("has_recognition_metrics", False):
            trend = analysis["recognition_trend"]
            report_lines.extend(
                [
                    "## Recognition Metrics Summary",
                    "",
                    "**Note:** Performance profiler metrics were not logged; report focuses on recognition quality trends.",
                    f"**Trend Source:** `{trend['source']}`",
                    "",
                    "| Metric | Best | Final | Delta |",
                    "|--------|------|-------|-------|",
                    f"| **val/acc** | {trend['best_val_acc']:.4f} | {trend['final_val_acc']:.4f} | {trend['final_val_acc'] - trend['best_val_acc']:+.4f} |",
                    f"| **val/cer** | {trend['best_val_cer']:.4f} | {trend['final_val_cer']:.4f} | {trend['final_val_cer'] - trend['best_val_cer']:+.4f} |",
                    f"| **val_loss** | {trend['min_val_loss']:.4f} | {trend['final_val_loss']:.4f} | {trend['final_val_loss'] - trend['min_val_loss']:+.4f} |",
                    "",
                    "| Training Details | Value |",
                    "|------------------|-------|",
                    f"| **Training Loss (final)** | {trend['final_train_loss']:.4f} |",
                    f"| **Validation Loss (final)** | {train_metrics['val_loss']:.4f} |",
                    f"| **Validation Accuracy (final)** | {train_metrics['val_acc']:.4f} |",
                    f"| **Validation CER (final)** | {train_metrics['val_cer']:.4f} |",
                    f"| **Epoch** | {train_metrics['epoch']} |",
                    f"| **Global Step** | {train_metrics['global_step']} |",
                    f"| **History Points (val/acc)** | {trend['history_points']} |",
                    "",
                ]
            )

            best_epoch_hint = trend.get("best_epoch_hint", float("nan"))
            best_epoch_total_hint = trend.get("best_epoch_total_hint", float("nan"))
            best_step_hint = trend.get("best_step_hint", float("nan"))
            best_steps_per_epoch_hint = trend.get("best_steps_per_epoch_hint", float("nan"))
            final_epoch_total_hint = trend.get("final_epoch_total_hint", float("nan"))
            final_epoch_hint = trend.get("final_epoch_hint", float("nan"))
            final_step_hint = trend.get("final_step_hint", float("nan"))
            final_steps_per_epoch_hint = trend.get("final_steps_per_epoch_hint", float("nan"))

            def _format_hint(epoch_hint: float, epoch_total_hint: float, step_hint: float, step_total_hint: float) -> str:
                if not math.isnan(epoch_hint):
                    epoch_str = f"epoch {int(epoch_hint)}/{int(epoch_total_hint)}" if not math.isnan(epoch_total_hint) and epoch_total_hint > 0 else f"epoch {int(epoch_hint)}"
                else:
                    epoch_str = "epoch N/A"

                if not math.isnan(step_hint):
                    step_str = f"step {int(step_hint)}/{int(step_total_hint)}" if not math.isnan(step_total_hint) and step_total_hint > 0 else f"step {int(step_hint)}"
                else:
                    step_str = "step N/A"

                if epoch_str == "epoch N/A" and step_str == "step N/A":
                    return "N/A"

                return f"{epoch_str}, {step_str}"

            report_lines.extend(
                [
                    "## Best/Final Snapshot Hints",
                    "",
                    f"- **Best val/acc point:** {_format_hint(best_epoch_hint, best_epoch_total_hint, best_step_hint, best_steps_per_epoch_hint)}",
                    f"- **Final val/acc point:** {_format_hint(final_epoch_hint, final_epoch_total_hint, final_step_hint, final_steps_per_epoch_hint)}",
                    "",
                ]
            )


            if trend["source"] == "local_output_log_regex":
                report_lines.extend(
                    [
                        "## Regex Parsing Diagnostics",
                        "",
                        "| Signal | Value |",
                        "|--------|-------|",
                        f"| **Raw metric lines found** | {int(trend['regex_raw_metric_lines'])} |",
                        f"| **Failed metric parses** | {int(trend['regex_parse_fail_lines'])} |",
                        f"| **Best debug match ratio** | {trend['regex_match_ratio_best']:.3f} |",
                        f"| **Final debug match ratio** | {trend['regex_match_ratio_final']:.3f} |",
                        "",
                    ]
                )

            if trend["lr_max"] > 0:
                report_lines.extend(
                    [
                        "## Learning Rate Snapshot",
                        "",
                        "| LR Metric | Value |",
                        "|----------|-------|",
                        f"| **LR Start** | {trend['lr_first']:.2e} |",
                        f"| **LR Peak** | {trend['lr_max']:.2e} |",
                        f"| **LR End** | {trend['lr_final']:.2e} |",
                        f"| **LR Min** | {trend['lr_min']:.2e} |",
                        "",
                    ]
                )

            report_lines.extend(
                [
                    "## Run Insights (Learning Guide)",
                    "",
                    "This run improved early, then regressed late. That usually means optimization overshot the best region rather than the model failing to learn.",
                    "",
                    "### Illustrated Interpretation",
                    "",
                    f"- **Best quality reached:** val/acc {trend['best_val_acc']:.4f}, val/cer {trend['best_val_cer']:.4f}",
                    f"- **End of run:** val/acc {trend['final_val_acc']:.4f}, val/cer {trend['final_val_cer']:.4f}",
                    f"- **Regression size:** Δacc {trend['final_val_acc'] - trend['best_val_acc']:+.4f}, Δcer {trend['final_val_cer'] - trend['best_val_cer']:+.4f}",
                    "",
                    "Simple mental model:",
                    "- Training = searching for a valley in error landscape.",
                    "- Best checkpoint = lowest spot reached so far.",
                    "- Late regression = optimizer steps moved away from that spot.",
                    "",
                ]
            )

            phase = analysis.get("comparisons", {}).get("recognition_phase")
            if phase:
                report_lines.extend(
                    [
                        "### Phase Trend",
                        "",
                        f"- **Early mean val/acc:** {phase['early_acc']:.4f}",
                        f"- **Middle mean val/acc:** {phase['mid_acc']:.4f}",
                        f"- **Late mean val/acc:** {phase['late_acc']:.4f}",
                        "",
                    ]
                )

            report_lines.extend(
                [
                    "## Hypothesis for Latest Regression",
                    "",
                    "Most plausible explanation is late-stage optimization instability during continuation from a strong checkpoint.",
                    "",
                    "- The model likely reached a good local optimum early in resumed training.",
                    "- Continued updates (and scheduler behavior) moved parameters away from that optimum.",
                    "- Validation noise from frequent in-epoch checks amplifies apparent fluctuations.",
                    "",
                ]
            )
        else:
            report_lines.extend(
                [
                    "## Training Metrics Summary",
                    "",
                    "**Note:** This run did not have performance profiling enabled. Showing available training metrics instead.",
                    "",
                    "| Metric | Validation | Test |",
                    "|--------|------------|------|",
                    f"| **H-Mean** | {train_metrics['val_hmean']:.4f} | {train_metrics['test_hmean']:.4f} |",
                    f"| **Precision** | {train_metrics['val_precision']:.4f} | {train_metrics['test_precision']:.4f} |",
                    f"| **Recall** | {train_metrics['val_recall']:.4f} | {train_metrics['test_recall']:.4f} |",
                    "",
                    "| Training Details | Value |",
                    "|------------------|-------|",
                    f"| **Training Loss** | {train_metrics['train_loss']:.4f} |",
                    f"| **Validation Loss** | {train_metrics['val_loss']:.4f} |",
                    f"| **Epoch** | {train_metrics['epoch']} |",
                    f"| **Global Step** | {train_metrics['global_step']} |",
                    "",
                ]
            )

    # Comparison with Training
    if analysis.get("comparisons", {}).get("validation_slowdown"):
        slowdown = analysis["comparisons"]["validation_slowdown"]
        report_lines.extend(
            [
                "## Training vs Validation Comparison",
                "",
                f"- **Validation Slowdown:** {slowdown['description']}",
                "",
            ]
        )
    else:
        if has_performance:
            report_lines.extend(
                [
                    "## Training vs Validation Comparison",
                    "",
                    "- **Note:** Training batch time not available in this run for comparison. Based on the performance plan, validation is typically ~10x slower than training due to PyClipper bottleneck.",
                    "",
                ]
            )
        else:
            report_lines.extend(
                [
                    "## Training vs Validation Comparison",
                    "",
                    "- **Note:** Performance profiling not enabled - cannot compare training vs validation timing.",
                    "",
                ]
            )

    # Bottlenecks
    report_lines.extend(
        [
            "## Identified Issues",
            "",
        ]
    )

    if analysis["bottlenecks"]:
        for i, bottleneck in enumerate(analysis["bottlenecks"], 1):
            report_lines.extend(
                [
                    f"### {i}. {bottleneck['type']} ({bottleneck['severity']})",
                    "",
                    f"{bottleneck['description']}",
                    "",
                ]
            )
    else:
        if has_performance:
            report_lines.append("No significant bottlenecks detected.")
        else:
            report_lines.append("No major issues detected in available metrics.")
        report_lines.append("")

    # Automatic next-run checklist for recognition runs
    if not has_performance and analysis.get("has_recognition_metrics", False):
        trend = analysis["recognition_trend"]
        acc_drop = trend["best_val_acc"] - trend["final_val_acc"]
        cer_rise = trend["final_val_cer"] - trend["best_val_cer"]

        def _gate_status(ok: bool, warn: bool = False) -> str:
            if ok:
                return "✅ PASS"
            if warn:
                return "⚠️ WARN"
            return "❌ FAIL"

        pre_lr_status = _gate_status(ok=acc_drop <= 0.01, warn=acc_drop <= 0.02)
        pre_val_status = _gate_status(ok=acc_drop <= 0.01, warn=acc_drop <= 0.03)
        inrun_acc_status = _gate_status(ok=acc_drop <= 0.01, warn=acc_drop <= 0.02)
        inrun_cer_status = _gate_status(ok=cer_rise <= 0.005, warn=cer_rise <= 0.015)
        post_ckpt_status = _gate_status(ok=acc_drop <= 0.0)
        post_continue_status = _gate_status(ok=acc_drop <= 0.01, warn=acc_drop <= 0.02)

        report_lines.extend(
            [
                "## Next Run Checklist (Auto-Gated)",
                "",
                "### Pre-run Gates",
                "",
                "| Gate | Status | Evidence | Action |",
                "|------|--------|----------|--------|",
                f"| **Resume LR conservative (<=2e-4)** | {pre_lr_status} | Δacc={-acc_drop:+.4f} from best to final | Lower LR by 5-10x for continuation runs. |",
                f"| **Validation cadence stable** | {pre_val_status} | Current run shows regression after interim peaks | Use `trainer.val_check_interval=1.0` for cleaner epoch-level signal. |",
                "",
                "### In-run Gates",
                "",
                "| Gate | Status | Evidence | Action |",
                "|------|--------|----------|--------|",
                f"| **No significant accuracy backslide** | {inrun_acc_status} | best={trend['best_val_acc']:.4f}, final={trend['final_val_acc']:.4f} | Early stop if `val/acc` drops >0.02 from run-best. |",
                f"| **CER remains near best** | {inrun_cer_status} | best={trend['best_val_cer']:.4f}, final={trend['final_val_cer']:.4f} | Reduce LR / halt when CER rises persistently. |",
                "",
                "### Post-run Gates",
                "",
                "| Gate | Status | Evidence | Action |",
                "|------|--------|----------|--------|",
                f"| **Inference checkpoint selection** | {post_ckpt_status} | Final underperformed best by {acc_drop:.4f} acc | Publish best-acc checkpoint, not final-epoch checkpoint. |",
                f"| **Continue training decision** | {post_continue_status} | Regression indicates optimization instability | Continue only with reduced LR + conservative scheduler. |",
                "",
            ]
        )

    # Additional Analysis
    if has_performance:
        report_lines.extend(
            [
                "## Additional Analysis",
                "",
                "Based on the performance optimization plan documented in the project handbook, the following issues are likely present:",
                "",
                "- **PyClipper Polygon Processing**: Known bottleneck causing ~10x validation slowdown",
                "- **Memory Usage**: Check for potential memory leaks during validation",
                "- **Batch Variance**: High variance in processing times indicating inconsistent performance",
                "",
            ]
        )
    else:
        if analysis.get("has_recognition_metrics", False):
            report_lines.extend(
                [
                    "## Recommendations",
                    "",
                    "1. **Select Best Checkpoint for Inference**: Use best `val/acc` checkpoint instead of final epoch checkpoint.",
                    "2. **Lower Resume LR for Fine-tuning**: For continuation runs, reduce LR by 5-10x to prevent post-resume regression.",
                    "3. **Stabilize Validation Signal**: Validate at epoch end (`trainer.val_check_interval=1.0`) for clearer epoch-level comparisons.",
                    "4. **Track LR Curves in W&B**: Ensure one LR key is logged continuously to correlate LR spikes with quality drops.",
                    "5. **Enable early-stop style guardrail**: Stop when best metric has not improved for N validations.",
                    "",
                    "Example continuation override:",
                    "```bash",
                    "uv run python scripts/runners/train.py mode=train experiment=parseq_flash_fast checkpoint_path=<best_ckpt> trainer.max_epochs=<target> train.optimizer.lr=2e-4 trainer.val_check_interval=1.0 train.logger.wandb.log_config=false",
                    "```",
                    "",
                    "Suggested scheduler stabilization (optional):",
                    "```bash",
                    "uv run python scripts/runners/train.py mode=train experiment=parseq_flash_fast checkpoint_path=<best_ckpt> trainer.max_epochs=<target> train.optimizer.lr=2e-4 train.lr_scheduler._target_=torch.optim.lr_scheduler.ReduceLROnPlateau train.lr_scheduler.mode=max train.lr_scheduler.monitor=val/acc train.lr_scheduler.factor=0.5 train.lr_scheduler.patience=3 train.lr_scheduler.min_lr=1e-6 trainer.val_check_interval=1.0",
                    "```",
                    "",
                ]
            )
        else:
            report_lines.extend(
                [
                    "## Recommendations",
                    "",
                    "To get detailed performance analysis, enable performance profiling in future runs:",
                    "",
                    "1. **Add Performance Profiler Callback**: Include `performance_profiler` in your training configuration",
                    "2. **Re-run Training**: Execute training with performance monitoring enabled",
                    "3. **Generate Full Report**: Use this script again on the profiled run",
                    "",
                    "Example config addition:",
                    "```yaml",
                    "callbacks:",
                    "  performance_profiler:",
                    "    _target_: ocr.core.lightning.callbacks.performance_profiler.PerformanceProfilerCallback",
                    "    enabled: true",
                    "    log_interval: 10",
                    "```",
                    "",
                ]
            )

    # Raw Metrics Summary
    report_lines.extend(
        [
            "## Raw Metrics Summary",
            "",
            "### Configuration",
            "```json",
            f"{json.dumps(metrics.get('config', {}), indent=2)}",
            "```",
            "",
            "### Summary Values",
            "```json",
            f"{json.dumps(metrics.get('summary', {}), indent=2)}",
            "```",
            "",
        ]
    )

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines))
    print(f"✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate baseline performance report from WandB run")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="WandB run ID",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="OCR_Performance_Baseline",
        help="WandB project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="Export raw metrics as JSON (optional)",
    )

    args = parser.parse_args()

    print(f"🔍 Fetching metrics from WandB run: {args.run_id}")
    metrics = fetch_wandb_metrics(args.run_id, args.project, args.entity)

    print("📊 Analyzing bottlenecks...")
    analysis = analyze_bottlenecks(metrics)

    print("📝 Generating markdown report...")
    generate_markdown_report(metrics, analysis, args.output)

    # Export JSON if requested
    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "analysis": analysis,
                },
                indent=2,
            )
        )
        print(f"✅ JSON export saved to: {args.export_json}")

    print("\n✨ Baseline report complete!")


if __name__ == "__main__":
    main()
