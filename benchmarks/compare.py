r"""Measure several trees interleaved and apply the port's no-slower statistic.

Usage::

    python benchmarks/compare.py --out-dir DIR --side main=<tree>/src --side branch=<tree>/src \\
        [--reference main] [--k 10] [--threads 1 8] [--cpus 0-7] [--filter REGEX]

The protocol (design/port_design.md, section 9.2, with the plan's DR-3): K rounds; in each round
every side runs once per thread setting, each in a fresh process pinned to ``--cpus`` with BLAS
and OpenMP threads at 1, the side order rotating from round to round so that no side always runs
first. Per process the statistic is the median of its repeats. Per side and case: the
**statistic** is the median of its K process medians and the **range** the max minus the min of
them. A side is flagged on a case when its statistic exceeds the reference's statistic plus the
reference's range; a flag counts only when a fresh interleaved re-run flags it again.

``--r6 ABI3,VERSIONED`` adds the stage-0 stable-ABI verdict on the ``tiny`` cases: the trigger is
an ABI3 statistic over the reference's statistic plus its range **and** more than the reference's
range above VERSIONED's statistic.

Writes ``raw/`` (every process's record), ``summary.json`` and ``summary.md`` under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

RUN = Path(__file__).resolve().parent / "run.py"


def _env(src: Path, threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    env["NUMBA_NUM_THREADS"] = str(max(threads, 1))
    return env


def _run(side: str, src: Path, threads: int, round_: int, args: argparse.Namespace) -> Path:
    out: Path = args.out_dir / "raw" / f"{side}.t{threads}.k{round_}.json"
    command = [
        "taskset",
        "-c",
        args.cpus,
        sys.executable,
        str(RUN),
        "--out",
        str(out),
        "--threads",
        str(threads),
        "--filter",
        args.filter,
        "--repeats",
        str(args.repeats),
    ]
    subprocess.run(command, env=_env(src, threads), check=True)
    record = json.loads(out.read_text())
    if not Path(record["pantr_file"]).is_relative_to(src):
        raise RuntimeError(f"side {side} imported {record['pantr_file']}, not under {src}")
    return out


def _summarize(records: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for (side, threads), recs in records.items():
        for name in recs[0]["cases"]:
            medians = [r["cases"][name]["median_ns"] for r in recs]
            entry = summary.setdefault(f"{name}@t{threads}", {"what": {}, "sides": {}})
            entry["what"][side] = recs[0]["cases"][name]["what"]
            entry["sides"][side] = {
                "statistic_ns": statistics.median(medians),
                "range_ns": max(medians) - min(medians),
                "process_medians_ns": medians,
            }
    return summary


def _flags(summary: dict[str, Any], reference: str, labels: list[str]) -> None:
    """Flag every side above the reference's bound; a case some side lacks is an error.

    Raises:
        ValueError: The reference is not a side, or a case is missing from a side (it would
            otherwise pass unflagged).
    """
    if reference not in labels:
        raise ValueError(f"reference {reference!r} is not one of the sides {labels}")
    for key, entry in summary.items():
        missing = [lab for lab in labels if lab not in entry["sides"]]
        if missing:
            raise ValueError(f"case {key} was not measured on {missing}")
        ref = entry["sides"][reference]
        bound = ref["statistic_ns"] + ref["range_ns"]
        for side, s in entry["sides"].items():
            if side != reference:
                s["flagged"] = s["statistic_ns"] > bound


def _r6(summary: dict[str, Any], reference: str, abi3: str, versioned: str) -> dict[str, Any]:
    verdicts: dict[str, Any] = {}
    for key, entry in summary.items():
        if not key.startswith("r6."):
            continue
        s = entry["sides"]
        ref, a, v = s[reference], s[abi3], s[versioned]
        over_main = a["statistic_ns"] > ref["statistic_ns"] + ref["range_ns"]
        over_versioned = a["statistic_ns"] - v["statistic_ns"] > ref["range_ns"]
        verdicts[key] = {
            "abi3_over_main_bound": over_main,
            "abi3_minus_versioned_over_main_range": over_versioned,
            "trigger": over_main and over_versioned,
        }
    return verdicts


def _markdown(summary: dict[str, Any], sides: list[str], reference: str) -> str:
    head = "| case | " + " | ".join(f"{s} stat (ns) | {s} range" for s in sides) + " | flags |"
    rule = "|" + "---|" * (2 * len(sides) + 2)
    lines = [head, rule]
    for key in sorted(summary):
        e = summary[key]["sides"]
        cells = []
        for s in sides:
            cells += (
                [f"{e[s]['statistic_ns']:.0f}", f"{e[s]['range_ns']:.0f}"] if s in e else ["", ""]
            )
        flags = ",".join(s for s in sides if s != reference and e.get(s, {}).get("flagged"))
        lines.append(f"| {key} | " + " | ".join(cells) + f" | {flags} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the interleaved protocol and write the summary.

    Returns:
        int: 0, or 1 when a side was flagged (or the R6 trigger fired). A failed measurement
        raises instead (a traceback and exit status 1 from the interpreter).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--side", action="append", required=True, help="label=absolute src path")
    parser.add_argument("--reference", default="main")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--cpus", default="0-7")
    parser.add_argument("--filter", default=".")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--r6", default=None, help="ABI3,VERSIONED side labels")
    args = parser.parse_args()

    sides = [(lab, Path(path).resolve()) for lab, path in (s.split("=", 1) for s in args.side)]
    records: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for round_ in range(args.k):
        order = sides[round_ % len(sides) :] + sides[: round_ % len(sides)]
        for threads in args.threads:
            for label, src in order:
                out = _run(label, src, threads, round_, args)
                records.setdefault((label, threads), []).append(json.loads(out.read_text()))
        print(f"round {round_ + 1}/{args.k} done", flush=True)

    summary = _summarize(records)
    _flags(summary, args.reference, [lab for lab, _ in sides])
    labels = [lab for lab, _ in sides]
    result: dict[str, Any] = {
        "protocol": {
            "k": args.k,
            "repeats": args.repeats,
            "threads": args.threads,
            "cpus": args.cpus,
            "reference": args.reference,
            "argv": sys.argv,
        },
        "sides": {
            lab: {
                "src": str(src),
                "commit": records[(lab, args.threads[0])][0]["commit"],
                "extension": records[(lab, args.threads[0])][0]["extension"],
            }
            for lab, src in sides
        },
        "machine": records[(labels[0], args.threads[0])][0]["machine"],
        "cases": summary,
    }
    flagged = any(s.get("flagged") for e in summary.values() for s in e["sides"].values())
    md = _markdown(summary, labels, args.reference)
    if args.r6:
        abi3, versioned = args.r6.split(",")
        # The verdict means something only if the labels name what they claim to.
        ext_of = {lab: result["sides"][lab]["extension"] for lab in (abi3, versioned)}
        if ".abi3." not in ext_of[abi3] or ".abi3." in ext_of[versioned]:
            raise ValueError(f"--r6 {args.r6}: the sides' modules are {ext_of}")
        result["r6"] = _r6(summary, args.reference, abi3, versioned)
        md += "\n| r6 case | abi3 > main stat + range | abi3 - versioned > main range | trigger |\n"
        md += "|---|---|---|---|\n"
        for key, v in sorted(result["r6"].items()):
            md += (
                f"| {key} | {v['abi3_over_main_bound']} | "
                f"{v['abi3_minus_versioned_over_main_range']} | {v['trigger']} |\n"
            )
        flagged = flagged or any(v["trigger"] for v in result["r6"].values())
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=1))
    (args.out_dir / "summary.md").write_text(md)
    print(md)
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
