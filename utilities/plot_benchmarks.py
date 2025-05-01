#!/usr/bin/env python3
"""
plot_benchmarks.py

Usage:
    python3 plot_benchmarks.py gpu1.txt gpu2.txt ...

Reads:
  - First line of each file as version header (must match)
  - Remaining lines as CSV with columns:
      benchmark, gpu, build, threads, section, runtime
Checks:
  - All version headers are identical
  - All benchmark sets are identical
Plots:
  - One PNG per benchmark, Debug/Release subplots
  - If multiple thread counts present, use only runs with thread count == 1
Outputs to the directory of the first input file.
"""

import os
import re
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def slugify(text):
    return re.sub(r'\W+', '_', text).strip('_')


def main():
    p = argparse.ArgumentParser(description="Plot benchmark runtimes")
    p.add_argument("reports", nargs='+',
                   help="Paths to the per-GPU report text files")
    args = p.parse_args()

    # Resolve & check input files
    report_paths = [os.path.abspath(r) for r in args.reports]
    for pth in report_paths:
        if not os.path.isfile(pth):
            sys.exit(f"ERROR: File not found: {pth}")

    # Output directory = directory of the first report
    out_dir = os.path.dirname(report_paths[0])

    versions = []
    dfs = []
    bench_sets = []

    # Read each file: capture version header and data
    for rp in report_paths:
        with open(rp, 'r') as f:
            versions.append(f.readline().strip())
        df = pd.read_csv(
            rp, skiprows=1,
            names=["benchmark","gpu","build","threads","section","runtime"],
            skipinitialspace=True
        )
        dfs.append(df)
        bench_sets.append(set(df["benchmark"].unique()))

    # 1) Validate that all versions match
    if len(set(versions)) != 1:
        sys.exit(f"ERROR: Version headers differ: {set(versions)}")
    version = versions[0]

    # 2) Validate that all benchmark sets match
    unique_sets = set(frozenset(s) for s in bench_sets)
    if len(unique_sets) != 1:
        msg = ["ERROR: Benchmark names differ across files:"]
        for rp, s in zip(report_paths, bench_sets):
            msg.append(f"  {os.path.basename(rp)}: {sorted(s)}")
        sys.exit("\n".join(msg))
    benchmarks = bench_sets[0]

    # 3) Merge all dataframes
    df_all = pd.concat(dfs, ignore_index=True)

    # 4) If multiple threads values present and thread '1' exists, filter to thread==1
    threads_unique = df_all['threads'].unique()
    if len(threads_unique) > 1 and 1 in threads_unique:
        df_all = df_all[df_all['threads'] == 1]

    # 5) Derive global section order from first appearance
    section_order = list(dict.fromkeys(df_all['section'].tolist()))

    # 6) Plot per benchmark
    for bench in benchmarks:
        grp = df_all[df_all['benchmark'] == bench]
        slug = slugify(bench)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        fig.suptitle(f"{bench} Performance ({version})", y=0.99)

        legend_h, legend_l = None, None

        # --- DEBUG subplot (capture legend) ---
        ax0 = axes[0]
        dbg = grp[grp.build == "Debug"]
        if not dbg.empty:
            tbl0 = dbg.pivot_table(index="gpu", columns="section",
                                   values="runtime", aggfunc="sum")
            # reorder columns
            cols0 = [s for s in section_order if s in tbl0.columns]
            tbl0 = tbl0[cols0]
            ax0 = tbl0.plot(kind="bar", stacked=True, ax=ax0, legend=True)
            handles, labels = ax0.get_legend_handles_labels()
            legend_h, legend_l = handles, labels
            ax0.get_legend().remove()
        else:
            ax0.axis('off')
        ax0.set_title("Debug")
        ax0.set_xlabel("GPU")
        ax0.set_ylabel("Runtime (s)")

        # --- RELEASE subplot ---
        ax1 = axes[1]
        rel = grp[grp.build == "Release"]
        if not rel.empty:
            tbl1 = rel.pivot_table(index="gpu", columns="section",
                                   values="runtime", aggfunc="sum")
            cols1 = [s for s in section_order if s in tbl1.columns]
            tbl1 = tbl1[cols1]
            tbl1.plot(kind="bar", stacked=True, ax=ax1, legend=False)
        else:
            ax1.axis('off')
        ax1.set_title("Release")
        ax1.set_xlabel("GPU")
        ax1.set_ylabel("Runtime (s)")

        # --- Shared legend just below title ---
        if legend_h:
            fig.legend(
                legend_h, legend_l,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.96),
                ncol=len(legend_l),
                frameon=False
            )

        fig.subplots_adjust(top=0.85)
        outpath = os.path.join(out_dir, f"{slug}.png")
        fig.savefig(outpath, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
