from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


def ensure_report_dirs(base_dir: str) -> Dict[str, str]:
    report_dir = os.path.join(base_dir, "report")
    imgs_dir = os.path.join(report_dir, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    return {
        "report_dir": report_dir,
        "imgs_dir": imgs_dir,
        # Final report (summary + analysis)
        "report_md": os.path.join(report_dir, "report.md"),
        # Experiment log (append-only)
        "exp_md": os.path.join(report_dir, "exp.md"),
    }


def append_markdown(report_md: str, md: str) -> None:
    os.makedirs(os.path.dirname(report_md), exist_ok=True)
    with open(report_md, "a", encoding="utf-8") as f:
        f.write(md)


def format_experiment_markdown(
    exp_name: str,
    dataset: str,
    source: str,
    target: str,
    config: Dict,
    results: Dict,
    notes: str = "",
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"\n\n## {exp_name}\n")
    lines.append(f"- time: {ts}\n")
    lines.append(f"- dataset: {dataset} ({source} → {target})\n")
    lines.append("- config:\n")
    for k, v in config.items():
        lines.append(f"  - {k}: {v}\n")
    lines.append("- results:\n")
    for k, v in results.items():
        lines.append(f"  - {k}: {v}\n")
    lines.append("- observations:\n")
    lines.append(f"  - {notes if notes else '[TODO]'}\n")
    return "".join(lines)
