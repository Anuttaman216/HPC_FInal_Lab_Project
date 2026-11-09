#!/usr/bin/env python3
"""analyze.py - derive speedup, efficiency and summary tables from the
benchmark CSV produced by scripts/run_benchmarks.sh.

Only measured data is used. Configurations are compared only with runs of the
same experiment label, grid and stopping criterion; missing baselines lead to
NaN metrics (and a warning), never to invented numbers.

Usage: python3 python/analyze.py [--results results]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = [
    "label", "mode", "comm", "nx", "ny", "nz", "rows", "nnz", "mpi_procs",
    "omp_threads", "processing_units", "iterations", "tolerance",
    "max_iterations", "converged", "time_median_s", "gflops",
    "comm_exchange_median_s", "comm_reduction_median_s", "comm_fraction",
    "time_per_iter_ms",
]

# Bytes moved per CG iteration under the streaming model documented in
# report/methodology.md: 12 B per non-zero (8 B value + 4 B column index)
# plus 96 B per row (row_ptr, gathered p, Ap write, vector updates).
BYTES_PER_NNZ = 12.0
BYTES_PER_ROW = 96.0

SERIES_ORDER = ["Sequential", "OpenMP", "MPI (allgather)", "MPI (halo)", "Hybrid (halo)", "Hybrid (allgather)"]


def config_name(row: pd.Series) -> str:
    """Human readable configuration name, e.g. 'MPI-halo 4p' or 'Hybrid 2x4'."""
    m, c, p, t = row["mode"], row["comm"], int(row["mpi_procs"]), int(row["omp_threads"])
    if m == "sequential":
        return "Sequential"
    if m == "openmp":
        return f"OpenMP {t}t"
    if m == "mpi":
        return f"MPI-{c} {p}p"
    return f"Hybrid-{c} {p}x{t}"


def series_name(row: pd.Series) -> str:
    """Implementation family (stable identity used for colours in graphs)."""
    m, c = row["mode"], row["comm"]
    if m in ("sequential", "openmp"):
        return {"sequential": "Sequential", "openmp": "OpenMP"}[m]
    return f"{'MPI' if m == 'mpi' else 'Hybrid'} ({c})"


def load(results: Path) -> pd.DataFrame:
    csv = results / "benchmark.csv"
    if not csv.exists() or csv.stat().st_size == 0:
        raise FileNotFoundError(f"{csv} is missing or empty - run scripts/run_benchmarks.sh first")
    df = pd.read_csv(csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{csv} lacks columns: {missing}")
    df["grid"] = df["nx"].astype(str) + "x" + df["ny"].astype(str) + "x" + df["nz"].astype(str)
    df["config"] = df.apply(config_name, axis=1)
    df["series"] = df.apply(series_name, axis=1)
    df["comm_time_s"] = df["comm_exchange_median_s"] + df["comm_reduction_median_s"]
    # Arithmetic intensity of one CG iteration (flops / modelled bytes).
    flops_it = 2.0 * df["nnz"] + 10.0 * df["rows"]
    bytes_it = BYTES_PER_NNZ * df["nnz"] + BYTES_PER_ROW * df["rows"]
    df["arith_intensity"] = flops_it / bytes_it
    df["effective_GBps"] = df["gflops"] / df["arith_intensity"]
    # If the same configuration was measured more than once, keep the latest.
    key = ["label", "mode", "comm", "grid", "mpi_procs", "omp_threads", "tolerance", "max_iterations"]
    df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    bad = df[df["converged"] != 1]
    if not bad.empty:
        print(f"warning: {len(bad)} configuration(s) did not converge and are excluded:", file=sys.stderr)
        print(bad[["label", "config", "grid", "status"]].to_string(index=False), file=sys.stderr)
    return df[df["converged"] == 1].copy()


def add_scaling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Speedup S = T_seq / T and efficiency E = S / units * 100, relative to the
    sequential run with the same label, grid and stopping criterion."""
    df = df.copy()
    df["speedup"] = np.nan
    df["efficiency_pct"] = np.nan
    group_keys = ["label", "grid", "tolerance", "max_iterations"]
    for _, g in df.groupby(group_keys):
        seq = g[g["mode"] == "sequential"]
        if seq.empty:
            continue
        t_seq = float(seq["time_median_s"].iloc[0])
        s = t_seq / g["time_median_s"]
        df.loc[g.index, "speedup"] = s
        df.loc[g.index, "efficiency_pct"] = 100.0 * s / g["processing_units"]
    return df


def weak_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Weak-scaling efficiency E_w(p) = T(1) / T(p) * 100 per implementation
    family (fixed work per unit and fixed iteration count)."""
    w = df[df["label"] == "weak"].copy()
    if w.empty:
        return w
    w["weak_efficiency_pct"] = np.nan
    for _, g in w.groupby("series"):
        base = g[g["processing_units"] == 1]
        if base.empty:
            continue
        t1 = float(base["time_median_s"].iloc[0])
        w.loc[g.index, "weak_efficiency_pct"] = 100.0 * t1 / g["time_median_s"]
    return w.sort_values(["series", "processing_units"])


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["label", "series", "config", "grid", "rows", "nnz", "mpi_procs", "omp_threads",
            "processing_units", "iterations", "relative_residual", "time_median_s",
            "time_min_s", "time_max_s", "time_per_iter_ms", "speedup", "efficiency_pct",
            "gflops", "effective_GBps", "comm_time_s", "comm_fraction", "memory_mb"]
    cols = [c for c in cols if c in df.columns]
    order = {"strong": 0, "size": 1, "weak": 2}
    out = df[cols].copy()
    family = {s: i for i, s in enumerate(SERIES_ORDER)}
    out["_o"] = out["label"].map(order).fillna(9)
    out["_f"] = out["series"].map(family).fillna(99)
    out = out.sort_values(["_o", "rows", "_f", "processing_units"]).drop(columns=["_o", "_f"])
    return out


def markdown_table(df: pd.DataFrame, columns: list[str], floatfmt: dict[str, str]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n"
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if np.isnan(v) else format(v, floatfmt.get(c, ".3f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = ap.parse_args()
    results = Path(args.results)

    try:
        df = load(results)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    df = add_scaling_metrics(df)
    summary = build_summary(df)
    summary.to_csv(results / "summary.csv", index=False, float_format="%.6g")

    fmt = {"time_median_s": ".4f", "speedup": ".2f", "efficiency_pct": ".1f", "gflops": ".2f",
           "comm_fraction": ".3f", "effective_GBps": ".1f", "time_per_iter_ms": ".3f"}
    md = ["# Benchmark summary (generated by python/analyze.py from results/benchmark.csv)\n"]
    strong = summary[summary["label"] == "strong"]
    if not strong.empty:
        g = strong["grid"].iloc[0]
        md.append(f"\n## Strong scaling (grid {g}, rows {int(strong['rows'].iloc[0])}, "
                  f"nnz {int(strong['nnz'].iloc[0])})\n\n")
        md.append(markdown_table(strong, ["config", "processing_units", "iterations", "time_median_s",
                                          "speedup", "efficiency_pct", "gflops", "effective_GBps",
                                          "comm_fraction"], fmt))
    size = summary[summary["label"] == "size"]
    if not size.empty:
        md.append("\n## Performance vs. matrix size\n\n")
        md.append(markdown_table(size, ["grid", "rows", "config", "iterations", "time_median_s",
                                        "speedup", "gflops", "effective_GBps"], fmt))
    weak = weak_scaling(df)
    if not weak.empty:
        weak.to_csv(results / "weak_scaling.csv", index=False, float_format="%.6g")
        md.append("\n## Weak scaling (fixed iterations, constant rows per unit)\n\n")
        md.append(markdown_table(weak, ["config", "grid", "rows", "processing_units", "time_median_s",
                                        "time_per_iter_ms", "weak_efficiency_pct", "gflops"],
                                 {**fmt, "weak_efficiency_pct": ".1f"}))
    bw = results / "bandwidth.csv"
    if bw.exists():
        b = pd.read_csv(bw)
        md.append("\n## Measured memory bandwidth (STREAM-style triad)\n\n")
        md.append(markdown_table(b, list(b.columns), {"best_GBps": ".1f", "median_GBps": ".1f"}))
    (results / "summary.md").write_text("".join(md), encoding="utf-8")

    print("".join(md))
    print(f"wrote {results / 'summary.csv'} and {results / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
