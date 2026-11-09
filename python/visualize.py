#!/usr/bin/env python3
"""visualize.py - performance graphs from the measured benchmark CSV.

Every figure is drawn from results/benchmark.csv (plus bandwidth.csv and the
convergence histories). Figures whose data is missing are skipped with a
message; nothing is ever filled with placeholder values.

Usage: python3 python/visualize.py [--results results]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

# Fixed colour per implementation family (identity never depends on rank or
# on which series happen to be present). Marker shape is a second encoding so
# identity does not rely on colour alone.
STYLE = {
    "Sequential":        dict(color="#52514e", marker="s"),
    "OpenMP":            dict(color="#2a78d6", marker="o"),
    "MPI (allgather)":   dict(color="#eb6834", marker="^"),
    "MPI (halo)":        dict(color="#1baf7a", marker="v"),
    "Hybrid (halo)":     dict(color="#c98500", marker="D"),
    "Hybrid (allgather)": dict(color="#6250d6", marker="P"),
}
ORDER = list(STYLE)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 7,
})

SUBTITLE = "HPCG-inspired CG (educational, not official HPCG)"


def save(fig, out: Path, name: str, made: list[str]) -> None:
    fig.tight_layout()
    path = out / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    made.append(path.name)
    print(f"  wrote graphs/{path.name}")


def title(ax, text: str, extra: str = "") -> None:
    ax.set_title(text + ("\n" + extra if extra else ""), loc="left", fontsize=11)


def plot_series(ax, df: pd.DataFrame, x: str, y: str) -> None:
    for s in ORDER:
        d = df[df["series"] == s].sort_values(x)
        if d.empty:
            continue
        if d[x].duplicated().any():
            # Several layouts share one x (e.g. hybrid 2x4 and 4x2 at p = 8):
            # a connecting line would be meaningless, so draw labelled markers.
            ax.plot(d[x], d[y], ls="none", label=s, **STYLE[s])
            for _, r in d.iterrows():
                ax.annotate(f"{int(r['mpi_procs'])}x{int(r['omp_threads'])}", (r[x], r[y]),
                            textcoords="offset points", xytext=(7, -3), fontsize=7.5, color=MUTED)
        else:
            ax.plot(d[x], d[y], label=s, **STYLE[s])


def log2_axis(ax, values) -> None:
    ax.set_xscale("log", base=2)
    ticks = sorted(set(int(v) for v in values))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = ap.parse_args()
    results = Path(args.results)
    out = results / "graphs"
    out.mkdir(parents=True, exist_ok=True)

    try:
        df = analyze.add_scaling_metrics(analyze.load(results))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    made: list[str] = []
    strong = df[df["label"] == "strong"]
    sgrid = strong["grid"].iloc[0] if not strong.empty else "?"
    seq_strong = strong[strong["mode"] == "sequential"]
    t_seq = float(seq_strong["time_median_s"].iloc[0]) if not seq_strong.empty else None
    info = f"grid {sgrid}, tol {strong['tolerance'].iloc[0]:g}, median of reps" if not strong.empty else ""

    # 1. Execution time vs threads (OpenMP) ---------------------------------
    omp = strong[strong["mode"] == "openmp"]
    if not omp.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        plot_series(ax, omp, "omp_threads", "time_median_s")
        if t_seq:
            ax.axhline(t_seq, color=STYLE["Sequential"]["color"], ls="--", lw=1.5, label="Sequential baseline")
        log2_axis(ax, omp["omp_threads"])
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("Solve time (s)")
        ax.set_ylim(bottom=0)
        title(ax, "1. Execution time vs. number of threads", info)
        ax.legend()
        save(fig, out, "01_time_vs_threads.png", made)
    else:
        print("  skip 01: no OpenMP strong-scaling data")

    # 2. Execution time vs MPI processes ------------------------------------
    mpi = strong[strong["mode"] == "mpi"]
    if not mpi.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        plot_series(ax, mpi, "mpi_procs", "time_median_s")
        if t_seq:
            ax.axhline(t_seq, color=STYLE["Sequential"]["color"], ls="--", lw=1.5, label="Sequential baseline")
        log2_axis(ax, mpi["mpi_procs"])
        ax.set_xlabel("MPI processes (1 thread each)")
        ax.set_ylabel("Solve time (s)")
        ax.set_ylim(bottom=0)
        title(ax, "2. Execution time vs. number of MPI processes", info)
        ax.legend()
        save(fig, out, "02_time_vs_mpi_procs.png", made)
    else:
        print("  skip 02: no MPI strong-scaling data")

    par = strong[strong["mode"] != "sequential"].dropna(subset=["speedup"])
    if not par.empty:
        # 3. Speedup ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        units = sorted(par["processing_units"].unique())
        ax.plot(units, units, color=MUTED, ls=":", lw=1.5, label="Ideal (S = p)")
        plot_series(ax, par, "processing_units", "speedup")
        log2_axis(ax, units)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Processing units p (MPI processes x OpenMP threads)")
        ax.set_ylabel("Speedup S(p) = T_seq / T(p)")
        title(ax, "3. Speedup vs. processing units", info)
        ax.legend(fontsize=9)
        save(fig, out, "03_speedup.png", made)

        # 4. Efficiency -------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.axhline(100, color=MUTED, ls=":", lw=1.5, label="Ideal (100 %)")
        plot_series(ax, par, "processing_units", "efficiency_pct")
        log2_axis(ax, units)
        ax.set_ylim(0, max(110, float(par["efficiency_pct"].max()) * 1.1))
        ax.set_xlabel("Processing units p")
        ax.set_ylabel("Parallel efficiency E(p) = S(p)/p (%)")
        title(ax, "4. Parallel efficiency vs. processing units", info)
        ax.legend(fontsize=9)
        save(fig, out, "04_efficiency.png", made)
    else:
        print("  skip 03/04: no parallel runs with a sequential baseline")

    # 5. GFLOPS comparison (all strong-scaling configurations) ---------------
    if not strong.empty:
        d = strong.copy()
        d["_o"] = d["series"].map({s: i for i, s in enumerate(ORDER)})
        d = d.sort_values(["_o", "processing_units"])
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        xs = np.arange(len(d))
        ax.bar(xs, d["gflops"], color=[STYLE[s]["color"] for s in d["series"]],
               width=0.8, edgecolor="white", linewidth=2)
        for x, v in zip(xs, d["gflops"]):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(xs)
        ax.set_xticklabels(d["config"], rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Estimated GFLOPS (model flops / median time)")
        ax.grid(axis="x", visible=False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=STYLE[s]["color"]) for s in ORDER if s in set(d["series"])]
        ax.legend(handles, [s for s in ORDER if s in set(d["series"])], fontsize=8, ncol=3)
        title(ax, "5. GFLOPS comparison of all configurations", info)
        save(fig, out, "05_gflops_comparison.png", made)

    # 6. Sequential vs OpenMP vs MPI vs Hybrid (best of each family) ---------
    if not strong.empty:
        best = strong.loc[strong.groupby("series")["time_median_s"].idxmin()].copy()
        best["_o"] = best["series"].map({s: i for i, s in enumerate(ORDER)})
        best = best.sort_values("_o")
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ys = np.arange(len(best))
        ax.barh(ys, best["time_median_s"], color=[STYLE[s]["color"] for s in best["series"]],
                height=0.6, edgecolor="white", linewidth=2)
        for y, (_, r) in zip(ys, best.iterrows()):
            sp = f", speedup {r['speedup']:.2f}x" if pd.notna(r["speedup"]) else ""
            ax.text(r["time_median_s"], y, f"  {r['time_median_s']:.3f} s ({r['config']}{sp})",
                    va="center", fontsize=8, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels(best["series"])
        ax.invert_yaxis()
        ax.set_xlabel("Best median solve time (s)")
        ax.set_xlim(0, float(best["time_median_s"].max()) * 1.9)
        ax.grid(axis="y", visible=False)
        title(ax, "6. Sequential vs. OpenMP vs. MPI vs. hybrid (best configuration)", info)
        save(fig, out, "06_implementation_comparison.png", made)

    # 7. Communication overhead vs MPI processes -----------------------------
    dist = strong[strong["mode"] == "mpi"].sort_values(["comm", "mpi_procs"])
    if not dist.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
        for s in ORDER:
            d = dist[dist["series"] == s]
            if d.empty:
                continue
            ax1.plot(d["mpi_procs"], 100 * d["comm_fraction"], label=s, **STYLE[s])
        log2_axis(ax1, dist["mpi_procs"])
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("MPI processes")
        ax1.set_ylabel("Communication share of solve time (%)")
        title(ax1, "7a. Communication overhead", "(exchange of p + Allreduce)")
        ax1.legend()

        labels = [f"{'AG' if c == 'allgather' else 'halo'} {p}p" for c, p in zip(dist["comm"], dist["mpi_procs"])]
        compute = dist["time_median_s"] - dist["comm_time_s"]
        xs = np.arange(len(dist))
        ax2.bar(xs, compute, color="#2a78d6", label="Computation (remaining time)", edgecolor="white", linewidth=1)
        ax2.bar(xs, dist["comm_exchange_median_s"], bottom=compute, color="#eb6834",
                label="Vector exchange (Allgatherv / halo)", edgecolor="white", linewidth=1)
        ax2.bar(xs, dist["comm_reduction_median_s"], bottom=compute + dist["comm_exchange_median_s"],
                color="#6250d6", label="MPI_Allreduce (incl. wait)", edgecolor="white", linewidth=1)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Time (s, rank average)")
        ax2.grid(axis="x", visible=False)
        title(ax2, "7b. Time breakdown per MPI configuration", info)
        ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        save(fig, out, "07_comm_overhead.png", made)
    else:
        print("  skip 07: no MPI data")

    # 8. Performance vs matrix size ------------------------------------------
    size = df[df["label"] == "size"]
    if not size.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
        for s in ORDER:
            d = size[size["series"] == s].sort_values("rows")
            if d.empty:
                continue
            lab = f"{s} ({d['config'].iloc[0].split()[-1]})" if s != "Sequential" else s
            ax1.plot(d["rows"], d["gflops"], label=lab, **STYLE[s])
            ax2.plot(d["rows"], d["time_per_iter_ms"], label=lab, **STYLE[s])
        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_xlabel("Matrix rows n (log scale)")
            ax.legend(fontsize=8)
        ax2.set_yscale("log")
        ax1.set_ylim(bottom=0)
        ax1.set_ylabel("Estimated GFLOPS")
        ax2.set_ylabel("Time per CG iteration (ms, log scale)")
        title(ax1, "8a. Performance vs. matrix size", "small problems fit in cache")
        title(ax2, "8b. Time per iteration vs. matrix size", "same tolerance for every size")
        save(fig, out, "08_perf_vs_size.png", made)
    else:
        print("  skip 08: no size-sweep data")

    # 9. Weak scaling ---------------------------------------------------------
    weak = analyze.weak_scaling(df)
    if not weak.empty:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.axhline(100, color=MUTED, ls=":", lw=1.5, label="Ideal (100 %)")
        plot_series(ax, weak, "processing_units", "weak_efficiency_pct")
        log2_axis(ax, weak["processing_units"])
        ax.set_ylim(0, 110)
        ax.set_xlabel("Processing units p (problem size grows with p)")
        ax.set_ylabel("Weak-scaling efficiency T(1)/T(p) (%)")
        base = weak["nx"].iloc[0]
        title(ax, "9. Weak scaling", f"{base}x{base}x({base}p) grid, fixed {int(weak['max_iterations'].iloc[0])} iterations")
        ax.legend(fontsize=9)
        save(fig, out, "09_weak_scaling.png", made)

    # 10. Convergence history -------------------------------------------------
    hist = sorted(results.glob("convergence_g*.csv"), key=lambda p: int(p.stem.split("g")[-1]))
    if hist:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
        for i, h in enumerate(hist):
            d = pd.read_csv(h)
            g = h.stem.split("g")[-1]
            ax.semilogy(d["iteration"], d["relative_residual"], color=colors[i % len(colors)],
                        lw=2, label=f"grid {g}^3 ({len(d) - 1} iterations)")
        ax.set_xlabel("CG iteration k")
        ax.set_ylabel("Relative residual ||r_k|| / ||b|| (log)")
        title(ax, "10. Convergence history (sequential CG)", "iterations grow with grid size (condition number)")
        ax.legend()
        save(fig, out, "10_convergence.png", made)

    # 11. Roofline (bandwidth roof x arithmetic intensity) --------------------
    bwf = results / "bandwidth.csv"
    if bwf.exists() and not strong.empty:
        bw = pd.read_csv(bwf)
        ai = float(strong["arith_intensity"].iloc[0])
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        xs = np.logspace(-1.5, 0.5, 100)
        bmax = float(bw["best_GBps"].max())
        b1 = float(bw.loc[bw["threads"].idxmin(), "best_GBps"])
        ax.loglog(xs, xs * bmax, color=MUTED, lw=1.5, ls="-", label=f"Memory roof, all threads ({bmax:.1f} GB/s triad)")
        ax.loglog(xs, xs * b1, color=MUTED, lw=1.5, ls="--", label=f"Memory roof, 1 thread ({b1:.1f} GB/s triad)")
        ax.axvline(ai, color=GRID, lw=1)
        ax.text(ai * 1.05, xs[0] * b1 * 1.5, f"CG intensity\n~{ai:.3f} flop/byte", fontsize=8, color=MUTED)
        for s in ORDER:
            d = strong[strong["series"] == s]
            if d.empty:
                continue
            best = d.loc[d["gflops"].idxmax()]
            ax.loglog([best["arith_intensity"]], [best["gflops"]], ls="none",
                      label=f"{s}: {best['gflops']:.2f} GFLOPS ({best['config']})", **STYLE[s])
        ax.set_xlabel("Arithmetic intensity (flop / byte, modelled)")
        ax.set_ylabel("Performance (GFLOPS)")
        title(ax, "11. Roofline view: CG is memory-bandwidth bound", info)
        ax.legend(fontsize=7.5, loc="upper left")
        save(fig, out, "11_roofline.png", made)

    (out / "INDEX.txt").write_text(
        "Graphs generated by python/visualize.py from measured data in results/:\n"
        + "\n".join(made) + "\n", encoding="utf-8")
    print(f"{len(made)} graphs written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
