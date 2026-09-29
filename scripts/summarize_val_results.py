from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = {
    "selected_ADE",
    "selected_FDE",
    "selected_collision",
    "selected_offroad",
    "selected_comfort",
    "score_regret",
}


def load_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of rows in {path}")
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["candidate_source"]), str(row["setting"]), str(row["method"])


def by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {row_key(row): row for row in rows}


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def rel_delta(candidate: float, baseline: float, metric: str) -> tuple[float, str]:
    delta = candidate - baseline
    pct = 100.0 * delta / max(abs(baseline), 1e-9)
    if metric in LOWER_IS_BETTER:
        direction = "better" if delta < 0 else "worse" if delta > 0 else "same"
    else:
        direction = "better" if delta > 0 else "worse" if delta < 0 else "same"
    return pct, direction


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def method_table(rows: list[dict[str, Any]], source: str, setting: str) -> str:
    selected = [
        row
        for row in rows
        if row.get("candidate_source") == source
        and row.get("setting") == setting
        and row.get("method")
        in {
            "proposal_first",
            "rule_current",
            "rule_based_cost",
            "no_rollout:no_rollout_scorer",
            "oracle_gt_score",
            "world:world",
            "world:proposal_aware",
            "world:balanced_robust",
            "world:robust",
            "world:robust_clean_preserve",
        }
    ]
    order = {
        "proposal_first": 0,
        "rule_current": 1,
        "rule_based_cost": 2,
        "no_rollout:no_rollout_scorer": 3,
        "world:world": 4,
        "world:proposal_aware": 5,
        "world:balanced_robust": 6,
        "world:robust": 7,
        "world:robust_clean_preserve": 8,
        "oracle_gt_score": 9,
    }
    selected.sort(key=lambda row: order.get(str(row["method"]), 99))
    table = [["method", "ADE", "FDE", "collision", "offroad", "progress", "regret"]]
    for row in selected:
        table.append(
            [
                str(row["method"]),
                fmt(float(row["selected_ADE"])),
                fmt(float(row["selected_FDE"])),
                fmt(float(row["selected_collision"])),
                fmt(float(row["selected_offroad"])),
                fmt(float(row["selected_progress"])),
                fmt(float(row["score_regret"])),
            ]
        )
    return markdown_table(table)


def comparison_table(rows: list[dict[str, Any]], source: str, baseline: str, candidates: list[str]) -> str:
    index = by_key(rows)
    table = [["setting", "method", "ADE", "FDE", "collision", "regret", "target_score"]]
    for setting in ["clean", "mild", "severe"]:
        base = index.get((source, setting, baseline))
        if base is None:
            continue
        for method in candidates:
            row = index.get((source, setting, method))
            if row is None:
                continue
            cells = [setting, method]
            for metric in ["selected_ADE", "selected_FDE", "selected_collision", "score_regret", "selected_target_score"]:
                pct, direction = rel_delta(float(row[metric]), float(base[metric]), metric)
                cells.append(f"{pct:+.1f}% {direction}")
            table.append(cells)
    return markdown_table(table)


def sanity_lines(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for source in sorted({str(row["candidate_source"]) for row in rows}):
        expert = [row for row in rows if row.get("candidate_source") == source and row.get("method") == "expert"]
        if not expert:
            continue
        clean = next((row for row in expert if row.get("setting") == "clean"), None)
        severe = next((row for row in expert if row.get("setting") == "severe"), None)
        if clean:
            out.append(
                f"- {source}: expert clean collision={float(clean['selected_collision']):.4f}, "
                f"offroad={float(clean['selected_offroad']):.4f}, valid={float(clean['valid_rate']):.4f}"
            )
        if clean and severe:
            dc = float(severe["selected_collision"]) - float(clean["selected_collision"])
            do = float(severe["selected_offroad"]) - float(clean["selected_offroad"])
            out.append(f"- {source}: expert severe-clean collision delta={dc:+.4f}, offroad delta={do:+.4f}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize validation JSON results into a paper-oriented Markdown report.")
    parser.add_argument("json_path")
    parser.add_argument("--candidate-source", default="proposal")
    parser.add_argument("--baseline", default="world:proposal_aware")
    parser.add_argument(
        "--compare",
        nargs="*",
        default=[
            "rule_current",
            "rule_based_cost",
            "no_rollout:no_rollout_scorer",
            "world:balanced_robust",
            "world:robust",
            "world:robust_clean_preserve",
        ],
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.json_path)
    parts = [
        f"# Validation Summary: {Path(args.json_path).name}",
        "",
        "## Sanity Checks",
        *sanity_lines(rows),
        "",
        f"## Main Table ({args.candidate_source})",
    ]
    for setting in ["clean", "mild", "severe"]:
        parts.extend(["", f"### {setting}", method_table(rows, args.candidate_source, setting)])
    parts.extend(
        [
            "",
            f"## Relative To {args.baseline}",
            "",
            comparison_table(rows, args.candidate_source, args.baseline, args.compare),
            "",
        ]
    )
    text = "\n".join(parts)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
