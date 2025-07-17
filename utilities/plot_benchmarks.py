#!/usr/bin/env python3
"""
plot_benchmarks.py

Usage:
    python3 plot_benchmarks.py gpu1.txt gpu2.txt ...

Reads:
  - First line of each file as version header (must match)
  - Remaining lines as CSV with columns:
      benchmark, gpu, cpu, build, threads, section, runtime
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


def is_text_file(filepath):
    """Check if a file is likely a text file by trying to read the first few bytes."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(100)  # Try to read first 100 characters
        return True
    except (UnicodeDecodeError, IOError):
        return False


def add_bar_annotations(ax, pivot_table):
    """Add total time annotations at the top of each bar."""
    totals = pivot_table.sum(axis=1)
    
    # Get the bar positions and heights
    bars = ax.patches
    x_positions = []
    heights = []
    
    # For stacked bars, we need to find the top of each stack
    n_bars = len(totals)
    n_sections = len(pivot_table.columns)
    
    for i in range(n_bars):
        # Find the topmost bar for this GPU
        top_bar_idx = i + (n_sections - 1) * n_bars
        if top_bar_idx < len(bars):
            bar = bars[top_bar_idx]
            x_pos = bar.get_x() + bar.get_width() / 2
            height = sum(pivot_table.iloc[i])  # Total height for this GPU
            
            # Add annotation
            ax.annotate(f'{height:.2f}s',
                       xy=(x_pos, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center',
                       va='bottom',
                       fontsize=9,
                       fontweight='bold')


def create_system_label(gpu, cpu):
    """Create a combined system label with GPU and CPU info."""
    # Truncate long names for better display
    gpu_short = gpu[:20] + "..." if len(gpu) > 23 else gpu
    cpu_short = cpu[:25] + "..." if len(cpu) > 28 else cpu
    return f"{gpu_short}\n{cpu_short}"


def write_markdown_report(df_all, version, benchmarks, section_order, out_dir):
    """Write a comprehensive markdown report of all benchmark data."""
    md_path = os.path.join(out_dir, "benchmark_report.md")
    
    with open(md_path, 'w') as f:
        # Header
        f.write(f"# Benchmark Performance Report\n\n")
        f.write(f"**Version:** {version}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System information
        f.write("## Test Systems\n\n")
        systems = df_all[['gpu', 'cpu']].drop_duplicates().sort_values('gpu')
        for _, row in systems.iterrows():
            f.write(f"- **GPU:** {row['gpu']}\n")
            f.write(f"  **CPU:** {row['cpu']}\n\n")
        
        # Summary statistics
        f.write("## Summary\n\n")
        f.write(f"- **Total Benchmarks:** {len(benchmarks)}\n")
        f.write(f"- **Systems Tested:** {len(systems)}\n")
        f.write(f"- **GPUs:** {', '.join(sorted(df_all['gpu'].unique()))}\n")
        f.write(f"- **Build Types:** {', '.join(sorted(df_all['build'].unique()))}\n")
        f.write(f"- **Sections:** {', '.join(section_order)}\n")
        f.write(f"- **Thread Count:** {df_all['threads'].iloc[0]}\n\n")
        
        # Detailed results per benchmark
        f.write("## Detailed Results\n\n")
        
        for bench in sorted(benchmarks):
            grp = df_all[df_all['benchmark'] == bench]
            f.write(f"### {bench}\n\n")
            
            # Debug results
            f.write("#### Debug Build\n\n")
            dbg = grp[grp.build == "Debug"]
            if not dbg.empty:
                # Create pivot table for Debug with system labels
                dbg_copy = dbg.copy()
                dbg_copy['system'] = dbg_copy['gpu'] + ' / ' + dbg_copy['cpu']
                tbl_dbg = dbg_copy.pivot_table(index="system", columns="section", 
                                        values="runtime", aggfunc="sum")
                cols_dbg = [s for s in section_order if s in tbl_dbg.columns]
                tbl_dbg = tbl_dbg[cols_dbg]
                
                # Add total column
                tbl_dbg['Total'] = tbl_dbg.sum(axis=1)
                
                # Write as markdown table
                f.write(tbl_dbg.to_markdown(floatfmt=".3f"))
                f.write("\n\n")
                
                # Performance breakdown
                f.write("**Debug Performance Breakdown:**\n\n")
                for system in tbl_dbg.index:
                    f.write(f"- **{system}:** {tbl_dbg.loc[system, 'Total']:.3f}s total\n")
                    for section in cols_dbg:
                        runtime = tbl_dbg.loc[system, section]
                        percentage = (runtime / tbl_dbg.loc[system, 'Total']) * 100
                        f.write(f"  - {section}: {runtime:.3f}s ({percentage:.1f}%)\n")
                f.write("\n")
            else:
                f.write("*No Debug data available*\n\n")
            
            # Release results
            f.write("#### Release Build\n\n")
            rel = grp[grp.build == "Release"]
            if not rel.empty:
                # Create pivot table for Release with system labels
                rel_copy = rel.copy()
                rel_copy['system'] = rel_copy['gpu'] + ' / ' + rel_copy['cpu']
                tbl_rel = rel_copy.pivot_table(index="system", columns="section", 
                                        values="runtime", aggfunc="sum")
                cols_rel = [s for s in section_order if s in tbl_rel.columns]
                tbl_rel = tbl_rel[cols_rel]
                
                # Add total column
                tbl_rel['Total'] = tbl_rel.sum(axis=1)
                
                # Write as markdown table
                f.write(tbl_rel.to_markdown(floatfmt=".3f"))
                f.write("\n\n")
                
                # Performance breakdown
                f.write("**Release Performance Breakdown:**\n\n")
                for system in tbl_rel.index:
                    f.write(f"- **{system}:** {tbl_rel.loc[system, 'Total']:.3f}s total\n")
                    for section in cols_rel:
                        runtime = tbl_rel.loc[system, section]
                        percentage = (runtime / tbl_rel.loc[system, 'Total']) * 100
                        f.write(f"  - {section}: {runtime:.3f}s ({percentage:.1f}%)\n")
                f.write("\n")
            else:
                f.write("*No Release data available*\n\n")
            
            # Debug vs Release comparison
            if not dbg.empty and not rel.empty:
                f.write("#### Debug vs Release Comparison\n\n")
                # Compare systems that appear in both builds
                dbg_systems = set(dbg['gpu'] + ' / ' + dbg['cpu'])
                rel_systems = set(rel['gpu'] + ' / ' + rel['cpu'])
                common_systems = dbg_systems & rel_systems
                
                for system in sorted(common_systems):
                    dbg_total = dbg_copy[dbg_copy['system'] == system]['runtime'].sum()
                    rel_total = rel_copy[rel_copy['system'] == system]['runtime'].sum()
                    speedup = dbg_total / rel_total
                    f.write(f"- **{system}:** {speedup:.2f}x speedup (Debug: {dbg_total:.3f}s → Release: {rel_total:.3f}s)\n")
                f.write("\n")
            
            f.write("---\n\n")
        
        # Overall performance comparison
        f.write("## Overall Performance Comparison\n\n")
        
        # System ranking by total performance
        f.write("### System Rankings\n\n")
        
        for build in sorted(df_all['build'].unique()):
            f.write(f"#### {build} Build Rankings\n\n")
            build_data = df_all[df_all['build'] == build].copy()
            build_data['system'] = build_data['gpu'] + ' / ' + build_data['cpu']
            system_totals = build_data.groupby('system')['runtime'].sum().sort_values()
            
            for rank, (system, total_time) in enumerate(system_totals.items(), 1):
                f.write(f"{rank}. **{system}**: {total_time:.3f}s total\n")
            f.write("\n")
        
        # Section performance analysis
        f.write("### Section Performance Analysis\n\n")
        for section in section_order:
            f.write(f"#### {section}\n\n")
            section_data = df_all[df_all['section'] == section].copy()
            section_data['system'] = section_data['gpu'] + ' / ' + section_data['cpu']
            
            for build in sorted(section_data['build'].unique()):
                build_section = section_data[section_data['build'] == build]
                if not build_section.empty:
                    f.write(f"**{build} Build:**\n\n")
                    system_times = build_section.groupby('system')['runtime'].sum().sort_values()
                    for system, time in system_times.items():
                        f.write(f"- {system}: {time:.3f}s\n")
                    f.write("\n")
        
        # Raw data appendix
        f.write("## Raw Data\n\n")
        f.write("### Complete Dataset\n\n")
        f.write(df_all.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n")
    
    print(f"Saved markdown report: {md_path}")


def main():
    p = argparse.ArgumentParser(description="Plot benchmark runtimes")
    p.add_argument("reports", nargs='+',
                   help="Paths to the per-GPU report text files")
    args = p.parse_args()

    # Resolve & check input files
    report_paths = [os.path.abspath(r) for r in args.reports]
    
    # Filter out non-existent files and non-text files
    valid_paths = []
    for pth in report_paths:
        if not os.path.isfile(pth):
            print(f"WARNING: File not found, skipping: {pth}")
            continue
        if not is_text_file(pth):
            print(f"WARNING: Not a text file, skipping: {pth}")
            continue
        valid_paths.append(pth)
    
    if not valid_paths:
        sys.exit("ERROR: No valid text files found")
    
    report_paths = valid_paths
    print(f"Processing {len(report_paths)} files: {[os.path.basename(p) for p in report_paths]}")

    # Output directory = directory of the first report
    out_dir = os.path.dirname(report_paths[0])

    versions = []
    dfs = []
    bench_sets = []

    # Read each file: capture version header and data
    for rp in report_paths:
        try:
            with open(rp, 'r') as f:
                versions.append(f.readline().strip())
            df = pd.read_csv(
                rp, skiprows=1,
                names=["benchmark","gpu","cpu","build","threads","section","runtime"],
                skipinitialspace=True
            )
            dfs.append(df)
            bench_sets.append(set(df["benchmark"].unique()))
        except Exception as e:
            sys.exit(f"ERROR: Failed to read file {rp}: {e}")

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

    # 6) Generate markdown report
    write_markdown_report(df_all, version, benchmarks, section_order, out_dir)

    # 7) Plot per benchmark
    for bench in benchmarks:
        grp = df_all[df_all['benchmark'] == bench]
        slug = slugify(bench)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        fig.suptitle(f"{bench} Performance ({version})", y=0.95)

        legend_h, legend_l = None, None

        # --- DEBUG subplot (capture legend) ---
        ax0 = axes[0]
        dbg = grp[grp.build == "Debug"]
        if not dbg.empty:
            # Create system labels for plotting
            dbg_copy = dbg.copy()
            dbg_copy['system'] = dbg_copy.apply(lambda row: create_system_label(row['gpu'], row['cpu']), axis=1)
            
            tbl0 = dbg_copy.pivot_table(index="system", columns="section",
                                   values="runtime", aggfunc="sum")
            # reorder columns
            cols0 = [s for s in section_order if s in tbl0.columns]
            tbl0 = tbl0[cols0]
            ax0 = tbl0.plot(kind="bar", stacked=True, ax=ax0, legend=True)
            handles, labels = ax0.get_legend_handles_labels()
            legend_h, legend_l = handles, labels
            ax0.get_legend().remove()
            
            # Add total time annotations
            add_bar_annotations(ax0, tbl0)
        else:
            ax0.axis('off')
        ax0.set_title("Debug Build")
        ax0.set_xlabel("System (GPU / CPU)")
        ax0.set_ylabel("Runtime (s)")
        ax0.tick_params(axis='x', rotation=45)

        # --- RELEASE subplot ---
        ax1 = axes[1]
        rel = grp[grp.build == "Release"]
        if not rel.empty:
            # Create system labels for plotting
            rel_copy = rel.copy()
            rel_copy['system'] = rel_copy.apply(lambda row: create_system_label(row['gpu'], row['cpu']), axis=1)
            
            tbl1 = rel_copy.pivot_table(index="system", columns="section",
                                   values="runtime", aggfunc="sum")
            cols1 = [s for s in section_order if s in tbl1.columns]
            tbl1 = tbl1[cols1]
            tbl1.plot(kind="bar", stacked=True, ax=ax1, legend=False)
            
            # Add total time annotations
            add_bar_annotations(ax1, tbl1)
        else:
            ax1.axis('off')
        ax1.set_title("Release Build")
        ax1.set_xlabel("System (GPU / CPU)")
        ax1.set_ylabel("Runtime (s)")
        ax1.tick_params(axis='x', rotation=45)

        # --- Shared legend just below title ---
        if legend_h:
            fig.legend(
                legend_h, legend_l,
                loc='upper center',
                bbox_to_anchor=(0.5, 0.91),
                ncol=min(len(legend_l), 4),  # Limit columns to prevent overcrowding
                frameon=False
            )

        fig.subplots_adjust(top=0.80, bottom=0.20)  # Adjust for rotated labels
        outpath = os.path.join(out_dir, f"{slug}.png")
        fig.savefig(outpath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()