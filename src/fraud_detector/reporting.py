"""Human-readable rendering of an evaluation report.

Evaluation numbers only change decisions if a human can read them at
a glance.  ``format_report`` prints the fraud-class metrics first,
the confusion matrix second, and the ranking metrics last — in that
order of decision relevance — with ``n/a`` wherever a ranking metric
could not be computed (single-class holdout, no probabilities).
"""

from __future__ import annotations

from fraud_detector.evaluation import EvaluationReport
from fraud_detector.thresholds import ThresholdChoice


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_report(
    model_name: str,
    report: EvaluationReport,
    threshold: ThresholdChoice | None = None,
) -> str:
    """Render ``report`` as a short text block for CLI/demo output."""
    lines = [
        f"model={model_name} rows={report.n} fraud={report.fraud_count}",
        f"  precision={report.precision:.4f} recall={report.recall:.4f} "
        f"f1={report.f1:.4f} accuracy={report.accuracy:.4f}",
        f"  confusion: TP={report.true_positives} FP={report.false_positives} "
        f"TN={report.true_negatives} FN={report.false_negatives}",
        f"  roc_auc={_metric(report.roc_auc)} pr_auc={_metric(report.pr_auc)}",
    ]
    if threshold is not None:
        lines.append(
            f"  best_f1_threshold={threshold.threshold:.2f} "
            f"(precision={threshold.precision:.4f} "
            f"recall={threshold.recall:.4f} f1={threshold.f1:.4f})"
        )
    return "\n".join(lines)
