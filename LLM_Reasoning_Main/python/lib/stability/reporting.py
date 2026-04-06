from __future__ import annotations

from typing import Any


def diagnostics_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Family Diagnostics",
        "",
        "| family | features | delta_f0.5 | delta_pr_auc | delta_p@10 | max_cross_corr |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family_id, payload in results.items():
        lines.append(
            "| {family} | {count} | {df05:+.4f} | {dpr:+.4f} | {dp10:+.4f} | {corr:.4f} |".format(
                family=family_id,
                count=int(payload["family_feature_count"]),
                df05=float(payload["delta_f0_5_mean"]),
                dpr=float(payload["delta_pr_auc_mean"]),
                dp10=float(payload["delta_precision_at_10_mean"]),
                corr=float(payload["cross_corr"]["max_abs_cross_corr"]),
            )
        )
    return "\n".join(lines)


def admission_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Sequential Admission",
        "",
        f"Admitted families: {', '.join(results['admitted_families']) if results['admitted_families'] else '(none)'}",
        "",
        "| family | accepted | delta_f0.5 | f0.5_std | delta_p@10 |",
        "|---|---|---:|---:|---:|",
    ]
    for decision in results["decisions"]:
        lines.append(
            "| {family} | {accepted} | {df05:+.4f} | {std:.4f} | {dp10:+.4f} |".format(
                family=decision["family_id"],
                accepted="yes" if decision["accepted"] else "no",
                df05=float(decision["delta_f0_5_mean"]),
                std=float(decision["delta_f0_5_std"]),
                dp10=float(decision["delta_precision_at_10_mean"]),
            )
        )
    return "\n".join(lines)


def routes_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Route Comparison",
        "",
        "| route | type | f0.5 | pr_auc | p@10 |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        summary = result["summary"]
        lines.append(
            "| {route_id} | {route_type} | {f05:.4f} | {pr:.4f} | {p10:.4f} |".format(
                route_id=result["route_id"],
                route_type=result["route_type"],
                f05=float(summary["f0_5_mean"]),
                pr=float(summary["pr_auc_mean"]),
                p10=float(summary["precision_at_10_mean"]),
            )
        )
    return "\n".join(lines)


def final_report_markdown(
    diagnostics: dict[str, Any],
    admission: dict[str, Any],
    routes: list[dict[str, Any]],
    residuals: dict[str, Any],
) -> str:
    lines = [
        "# Instability-Control Summary",
        "",
        "## Admission Outcome",
        "",
        f"Admitted families: {', '.join(admission['admitted_families']) if admission['admitted_families'] else '(none)'}",
        "",
        "## Best Route",
        "",
    ]
    if routes:
        best = max(routes, key=lambda item: float(item["summary"]["f0_5_mean"]))
        lines.append(
            f"{best['route_id']} ({best['route_type']}) with F0.5={float(best['summary']['f0_5_mean']):.4f}"
        )
    else:
        lines.append("No routes evaluated.")
    lines += [
        "",
        "## Residual Gain",
        "",
        f"Recovered false negatives: {residuals['recovered_false_negatives']} / {residuals['baseline_false_negatives']}",
        f"Borderline recovered: {residuals['borderline_recovered']} / {residuals['borderline_case_count']}",
        "",
        "## Diagnostics Snapshot",
        "",
        diagnostics_markdown(diagnostics),
    ]
    return "\n".join(lines)
