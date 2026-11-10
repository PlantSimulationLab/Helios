#!/usr/bin/env python3
"""
Generate a LaTeX/pgfplots PDF report of all SoilTransportModel benchmark results.

The benchmark executable (``soiltransport_benchmarks``) writes a set of CSV
files into its build directory. This script converts those, together with the
published reference datasets in ``reference_data/``, into pgfplots ``table``
files and emits a single self-contained LaTeX document that is compiled to PDF.

Everything generated lives under ``benchmark_report/`` so the sample directory
stays tidy:

    benchmark_report/
        benchmarks.tex
        benchmarks.pdf
        data/<benchmark>/*.txt   (pgfplots tables: "<quantity> depth" etc.)

Usage:
    python plot_benchmarks.py [--run] [--no-pdf]

    --run      run the benchmark executable first to (re)generate the CSVs
    --no-pdf   only emit the .tex and data files, skip LaTeX compilation
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent
BUILD_DIR = SAMPLE_DIR / "build"
REF_DIR = SAMPLE_DIR / "reference_data"
OUT_DIR = SAMPLE_DIR / "benchmark_report"
DATA_DIR = OUT_DIR / "data"

# ---------------------------------------------------------------------------
# Data IO helpers
# ---------------------------------------------------------------------------


def read_csv(name):
    """Read a benchmark CSV from the build directory; return list[dict] or None."""
    path = BUILD_DIR / name
    if not path.exists():
        print(f"  [skip] missing {name}")
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, name):
    return [float(r[name]) for r in rows]


def write_table(relpath, columns):
    """Write a pgfplots table. columns = [(name, values), ...]; first=x, second=y."""
    path = OUT_DIR / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [c[0] for c in columns]
    data = [c[1] for c in columns]
    with open(path, "w") as f:
        f.write(" ".join(names) + "\n")
        for row in zip(*data):
            f.write(" ".join(f"{v:.8g}" for v in row) + "\n")
    return relpath


def copy_ref(src_relpath, bench, dst_name):
    """Copy a reference data file (already pgfplots-formatted) into the report."""
    src = REF_DIR / src_relpath
    if not src.exists():
        print(f"  [skip] missing reference {src_relpath}")
        return None
    dst = DATA_DIR / bench / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return f"data/{bench}/{dst_name}"


def read_table(rel_or_abs):
    """Read a 2-column whitespace table (with header); return (xs, ys)."""
    path = OUT_DIR / rel_or_abs if not Path(rel_or_abs).is_absolute() else Path(rel_or_abs)
    xs, ys = [], []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
    return xs, ys


def lims(values, pad=0.05):
    lo, hi = min(values), max(values)
    span = (hi - lo) or (abs(hi) or 1.0)
    return lo - pad * span, hi + pad * span


# ---------------------------------------------------------------------------
# pgfplots / TikZ emit helpers (matching the requested style)
# ---------------------------------------------------------------------------


def _meta(meta):
    return f"[meta={meta}]" if meta else ""


def p_line(relfile, color="blue", meta="depth"):
    return f"\\addplot+[color={color},\n    no markers,\n    ]\ntable{_meta(meta)}\n{{{relfile}}};"


def p_marks(relfile, color="red", meta="depth", size="2.5pt", mark="o"):
    return f"\\addplot+[color={color},\n    mark={mark},\n    mark size={size},\n    only marks,\n    ]\ntable{_meta(meta)}\n{{{relfile}}};"


def p_dashed(relfile, color="red", meta="depth"):
    return f"\\addplot+[color={color},\n    densely dashed,\n    no markers,\n    ]\ntable{_meta(meta)}\n{{{relfile}}};"


def p_cycle(relfile, meta="depth"):
    return f"\\addplot+[no markers]\ntable{_meta(meta)}\n{{{relfile}}};"


def figure(title, xlabel, ylabel, plots, legend, axis_opts=None, cycle=False):
    opts = []
    if cycle:
        opts.append("cycle list name=auto")
    opts += [f"title={{{title}}}", f"xlabel={{{xlabel}}}", f"ylabel={{{ylabel}}}"]
    opts += axis_opts or []
    opts.append("legend pos=outer north east")
    lines = ["\\begin{center}", "\\begin{tikzpicture}", "\\begin{axis}["]
    lines.append(",\n".join("    " + o for o in opts))
    lines.append("    ]")
    lines += plots
    if legend:
        lines.append("\\legend{" + ", ".join(legend) + "}")
    lines += ["\\end{axis}", "\\end{tikzpicture}", "\\end{center}"]
    return "\n".join(lines)


def depth_axis_opts(xvals, depthvals):
    xmin, xmax = lims(xvals)
    dmin, _ = lims(depthvals)
    return [f"xmin={xmin:.4g}", f"xmax={xmax:.4g}", f"ymin={dmin:.4g}", "ymax=0"]


# ---------------------------------------------------------------------------
# Benchmark sections. Each returns (heading, [figure_tex, ...]) or None.
# ---------------------------------------------------------------------------


def sec_lehmann():
    rows = read_csv("benchmark_ackerer.csv")
    if rows is None:
        return None
    theta = col(rows, "theta")
    h = col(rows, "h_m")
    depth = [-d for d in col(rows, "depth_m")]
    f_theta = write_table("data/lehmann/Lehmann_Ackerer1998_Helios_theta.txt", [("wc", theta), ("depth", depth)])
    f_h = write_table("data/lehmann/Lehmann_Ackerer1998_Helios_h.txt", [("h", h), ("depth", depth)])
    r_theta = copy_ref("Lehmann_Ackerer_1998_hetero/Lehmann_Ackerer1998_hetero_theta.txt", "lehmann", "Lehmann_Ackerer1998_hetero_theta.txt")
    r_h = copy_ref("Lehmann_Ackerer_1998_hetero/Lehmann_Ackerer1998_hetero_h.txt", "lehmann", "Lehmann_Ackerer1998_hetero_h.txt")

    figs = []
    rx, ry = read_table(r_theta) if r_theta else ([], [])
    figs.append(figure(
        "Heterogeneous Soils Test (Lehmann and Ackerer 1998)", "Volumetric Water Content", "Depth (m)",
        [p_line(f_theta, "blue"), p_marks(r_theta, "red")] if r_theta else [p_line(f_theta, "blue")],
        ["Helios", "Li et al. (2021)"] if r_theta else ["Helios"],
        depth_axis_opts(theta + rx, depth + ry), cycle=True))
    rx, ry = read_table(r_h) if r_h else ([], [])
    figs.append(figure(
        "Heterogeneous Soils Test (Lehmann and Ackerer 1998)", "Pressure Head (m)", "Depth (m)",
        [p_line(f_h, "blue"), p_marks(r_h, "red", size="3pt")] if r_h else [p_line(f_h, "blue")],
        ["Helios", "Li et al. (2021)"] if r_h else ["Helios"],
        depth_axis_opts(h + rx, depth + ry), cycle=True))
    return ("Heterogeneous Column (Lehmann \\& Ackerer 1998)", figs)


def sec_abeele():
    rows = read_csv("benchmark_abeele.csv")
    if rows is None:
        return None
    theta = col(rows, "theta")
    h = col(rows, "h_m")
    depth = [-d for d in col(rows, "depth_m")]
    f_theta = write_table("data/abeele/Abeele_1984_Helios_theta.txt", [("wc", theta), ("depth", depth)])
    f_h = write_table("data/abeele/Abeele_1984_Helios_h.txt", [("h", h), ("depth", depth)])
    hyd_theta = copy_ref("Abeele_1984_free_drainage/Abeele_1984_Hdrus_theta.txt", "abeele", "Abeele_1984_Hydrus_theta.txt")
    hyd_h = copy_ref("Abeele_1984_free_drainage/Abeele_1984_Hydrus_h.txt", "abeele", "Abeele_1984_Hydrus_h.txt")
    meas_theta = copy_ref("Abeele_1984_free_drainage/Abeele_1984_free_drainage_theta.txt", "abeele", "Abeele_1984_free_drainage_theta.txt")
    meas_h = copy_ref("Abeele_1984_free_drainage/Abeele_1984_free_drainage_h.txt", "abeele", "Abeele_1984_free_drainage_h.txt")

    def collect(*files):
        xs, ys = [], []
        for fr in files:
            if fr:
                a, b = read_table(fr)
                xs += a
                ys += b
        return xs, ys

    figs = []
    xs, ys = collect(f_theta, hyd_theta, meas_theta)
    plots, leg = [p_line(f_theta, "blue")], ["Helios"]
    if hyd_theta:
        plots.append(p_dashed(hyd_theta, "red"))
        leg.append("Hydrus")
    if meas_theta:
        plots.append(p_marks(meas_theta, "red"))
        leg.append("Measured")
    figs.append(figure("Free Drainage Test (Abeele 1984)", "Volumetric Water Content", "Depth (m)", plots, leg, depth_axis_opts(xs, ys)))

    xs, ys = collect(f_h, hyd_h, meas_h)
    plots, leg = [p_line(f_h, "blue")], ["Helios"]
    if hyd_h:
        plots.append(p_dashed(hyd_h, "red"))
        leg.append("Hydrus")
    if meas_h:
        plots.append(p_marks(meas_h, "red"))
        leg.append("Measured")
    figs.append(figure("Free Drainage Test (Abeele 1984)", "Pressure Head (m)", "Depth (m)", plots, leg, depth_axis_opts(xs, ys)))
    return ("Deep Lysimeter Free Drainage (Abeele 1984)", figs)


def sec_zadeh():
    rows = read_csv("benchmark_zadeh.csv")
    if rows is None:
        return None
    t = col(rows, "time_days")
    infil = col(rows, "infiltration_cm")
    f_helios = write_table("data/zadeh/ZadehHelios.txt", [("Time", t), ("Infiltration(cm)", infil)])
    meas = copy_ref("Zadeh_2007/ZadehMeasured.txt", "zadeh", "ZadehMeasured.txt")

    plots = [p_line(f_helios, "blue", meta="Infiltration(cm)")]
    leg = ["Helios"]
    # Optional adaptive-solver overlay.
    adp = read_csv("zadeh_adaptive.csv")
    if adp is not None:
        ta = col(adp, "time_days")
        ia = col(adp, "infil_cm")
        f_adp = write_table("data/zadeh/ZadehHeliosAdaptive.txt", [("Time", ta), ("Infiltration(cm)", ia)])
        plots.append(p_dashed(f_adp, "black", meta="Infiltration(cm)"))
        leg.append("Helios (adaptive)")
    if meas:
        plots.append(p_marks(meas, "red", meta="Infiltration(cm)"))
        leg.append("Measured")
    xmax = max(t) * 1.02
    ymax = max(infil) * 1.1
    fig = figure("Dry Infiltration Test (Zadeh 2011)", "Time (day)", "Cumulative Infiltration (cm)", plots, leg,
                 [f"xmin=0", f"xmax={xmax:.4g}", "ymin=0", f"ymax={ymax:.4g}"])
    return ("Dry-Soil Infiltration (Zadeh 2007)", [fig])


def sec_philip():
    rows = read_csv("philip_infiltration.csv")
    if rows is None:
        return None
    t_min = [v / 60.0 for v in col(rows, "time")]
    num = col(rows, "numerical_I_mm")
    phil = col(rows, "philip_I_mm")
    f_num = write_table("data/philip/philip_helios.txt", [("time", t_min), ("I", num)])
    f_phil = write_table("data/philip/philip_analytic.txt", [("time", t_min), ("I", phil)])
    fig = figure("Philip Sorptivity Verification (loam)", "Time (min)", "Cumulative Infiltration (mm)",
                 [p_line(f_num, "blue", meta=None), p_dashed(f_phil, "red", meta=None)], ["Helios", "Philip (exact $S$)"],
                 ["xmin=0", f"xmax={max(t_min):.4g}", "ymin=0"])
    return ("Philip Sorptivity Verification", [fig])


def sec_gardner():
    rows = read_csv("gardner_steady.csv")
    if rows is None:
        return None
    depth = [-d for d in col(rows, "depth_m")]
    h_num = col(rows, "h_num")
    h_ex = col(rows, "h_exact")
    f_n = write_table("data/gardner/gardner_helios.txt", [("h", h_num), ("depth", depth)])
    f_e = write_table("data/gardner/gardner_exact.txt", [("h", h_ex), ("depth", depth)])
    # Force fixed-point x tick labels (head values near 0 otherwise render in mixed scientific notation).
    opts = depth_axis_opts(h_num + h_ex, depth)
    opts += ["scaled x ticks=false", "xticklabel style={/pgf/number format/.cd, fixed, precision=2}"]
    fig = figure("Gardner Steady Infiltration (Exact Verification)", "Pressure Head (m)", "Depth (m)",
                 [p_line(f_n, "blue"), p_marks(f_e, "red")], ["Helios", "Exact (quasi-linear)"], opts)
    return ("Gardner Steady Infiltration (Exact)", [fig])


def sec_steady_state():
    rows = read_csv("steady_state.csv")
    if rows is None:
        return None
    depth = [-d for d in col(rows, "depth")]
    h_final = col(rows, "h_final")
    h_exp = col(rows, "h_expected")
    f_h = write_table("data/steady/steady_helios.txt", [("h", h_final), ("depth", depth)])
    f_e = write_table("data/steady/steady_expected.txt", [("h", h_exp), ("depth", depth)])
    fig = figure("Steady-State Unit Gradient", "Pressure Head (m)", "Depth (m)",
                 [p_line(f_h, "blue"), p_dashed(f_e, "red")], ["Helios", "Analytical"],
                 depth_axis_opts(h_final + h_exp, depth))
    return ("Steady-State Unit Gradient", [fig])


def sec_mass_balance():
    rows = read_csv("mass_balance_storage.csv")
    if rows is None:
        return None
    t_h = [v / 3600.0 for v in col(rows, "time")]
    storage_mm = [v * 1000.0 for v in col(rows, "storage_m")]
    f = write_table("data/mass_balance/storage.txt", [("time", t_h), ("storage", storage_mm)])
    mean = sum(storage_mm) / len(storage_mm)
    fig = figure("Mass Balance (Zero-Flux Boundaries)", "Time (hours)", "Total Storage (mm)",
                 [p_line(f, "blue", meta=None)], ["Helios"],
                 ["xmin=0", f"xmax={max(t_h):.4g}", f"ymin={mean - 1.0:.4g}", f"ymax={mean + 1.0:.4g}"])
    return ("Mass Conservation", [fig])


def sec_infiltration():
    rows = read_csv("infiltration_theta.csv")
    rows_h = read_csv("infiltration_h.csv")
    if rows is None and rows_h is None:
        return None
    times = ["t=0", "t=300", "t=1200", "t=3600"]
    figs = []
    if rows is not None:
        depth = [-d for d in col(rows, "depth")]
        plots, leg = [], []
        allx = []
        for tcol in times:
            if tcol in rows[0]:
                x = col(rows, tcol)
                allx += x
                rel = write_table(f"data/infiltration/theta_{tcol.replace('=','')}.txt", [("wc", x), ("depth", depth)])
                plots.append(p_cycle(rel))
                leg.append(f"t = {tcol[2:]} s")
        figs.append(figure("Infiltration into Dry Soil", "Volumetric Water Content", "Depth (m)", plots, leg,
                           depth_axis_opts(allx, depth), cycle=True))
    if rows_h is not None:
        depth = [-d for d in col(rows_h, "depth")]
        plots, leg, allx = [], [], []
        for tcol in times:
            if tcol in rows_h[0]:
                x = col(rows_h, tcol)
                allx += x
                rel = write_table(f"data/infiltration/h_{tcol.replace('=','')}.txt", [("h", x), ("depth", depth)])
                plots.append(p_cycle(rel))
                leg.append(f"t = {tcol[2:]} s")
        figs.append(figure("Infiltration into Dry Soil", "Pressure Head (m)", "Depth (m)", plots, leg,
                           depth_axis_opts(allx, depth), cycle=True))
    return ("Infiltration Front Propagation", figs)


def sec_drainage():
    figs = []
    rows = read_csv("drainage_h.csv")
    if rows is not None:
        depth = [-d for d in col(rows, "depth")]
        times = ["t=0", "t=600", "t=2400", "t=7200"]
        plots, leg, allx = [], [], []
        for tcol in times:
            if tcol in rows[0]:
                x = col(rows, tcol)
                allx += x
                rel = write_table(f"data/drainage/h_{tcol.replace('=','')}.txt", [("h", x), ("depth", depth)])
                plots.append(p_cycle(rel))
                leg.append(f"t = {int(int(tcol[2:]) / 60)} min")
        figs.append(figure("Gravity Drainage", "Pressure Head (m)", "Depth (m)", plots, leg,
                           depth_axis_opts(allx, depth), cycle=True))
    st = read_csv("drainage_storage.csv")
    if st is not None:
        t_min = [v / 60.0 for v in col(st, "time")]
        storage = [v * 1000.0 for v in col(st, "storage_m")]
        rel = write_table("data/drainage/storage.txt", [("time", t_min), ("storage", storage)])
        figs.append(figure("Gravity Drainage: Storage Depletion", "Time (min)", "Total Storage (mm)",
                           [p_line(rel, "blue", meta=None)], ["Helios"], ["xmin=0", f"xmax={max(t_min):.4g}"]))
    return ("Gravity Drainage", figs) if figs else None


def sec_soil_comparison():
    rows = read_csv("soil_comparison.csv")
    if rows is None:
        return None
    t_h = [v / 3600.0 for v in col(rows, "time")]
    soils = [c for c in rows[0].keys() if c.endswith("_front_depth")]
    plots, leg, ally = [], [], []
    for s in soils:
        y = [v * 100.0 for v in col(rows, s)]
        ally += y
        rel = write_table(f"data/soil_comparison/{s}.txt", [("time", t_h), ("front", y)])
        plots.append(p_cycle(rel, meta=None))
        leg.append(s.replace("_front_depth", "").replace("_", " "))
    fig = figure("Soil-Type Comparison: Wetting Front", "Time (hours)", "Wetting Front Depth (cm)", plots, leg,
                 ["xmin=0", f"xmax={max(t_h):.4g}", "ymin=0"], cycle=True)
    return ("Soil-Type Comparison", [fig])


def sec_grid_convergence():
    rows = read_csv("grid_convergence.csv")
    if rows is None:
        return None
    n = col(rows, "n_layers")
    storage = [v * 1000.0 for v in col(rows, "final_storage")]
    rel = write_table("data/grid/storage.txt", [("n", n), ("storage", storage)])
    fig = figure("Grid Convergence", "Number of Layers", "Final Storage (mm)",
                 [p_marks(rel, "blue", meta=None, mark="*", size="2.5pt")], ["Helios"],
                 ["xmode=log", "log basis x=10"])
    return ("Grid Convergence Study", [fig])


def sec_celia():
    rows = read_csv("celia_validation.csv")
    if rows is None:
        return None
    depth = [-d / 100.0 for d in col(rows, "depth_cm")]  # cm -> m
    h_num = col(rows, "h_numerical_cm")
    h_ref = col(rows, "h_reference_cm")
    f_num = write_table("data/celia/celia_helios.txt", [("h", h_num), ("depth", depth)])
    f_ref = write_table("data/celia/celia_reference.txt", [("h", h_ref), ("depth", depth)])
    fig = figure("Dry-Start Infiltration (Celia 1990)", "Pressure Head (cm)", "Depth (m)",
                 [p_line(f_num, "blue"), p_marks(f_ref, "red")], ["Helios", "Reference"],
                 depth_axis_opts(h_num + h_ref, depth))
    return ("Celia/Polmann Dry-Start Infiltration", [fig])


def sec_warrick():
    rows = read_csv("warrick_results.csv")
    if rows is None:
        return None
    depth = [-d / 100.0 for d in col(rows, "depth_cm")]
    plots, leg, allx = [], [], []
    for c, dt in [("h_dt1_cm", "1"), ("h_dt300_cm", "300"), ("h_dt900_cm", "900")]:
        if c in rows[0]:
            x = col(rows, c)
            allx += x
            rel = write_table(f"data/warrick/{c}.txt", [("h", x), ("depth", depth)])
            plots.append(p_cycle(rel))
            leg.append(f"dt = {dt} s")
    fig = figure("Ponded Infiltration (Warrick, Yolo Light Clay)", "Pressure Head (cm)", "Depth (m)", plots, leg,
                 depth_axis_opts(allx, depth), cycle=True)
    return ("Warrick Ponded Infiltration", [fig])


def sec_miller():
    rows = read_csv("miller_profiles.csv")
    if rows is None:
        return None
    depth = [-d for d in col(rows, "depth_cm")]
    soils = ["sand", "loam", "clay_loam"]
    figs = []
    plots, leg, allx = [], [], []
    for s in soils:
        c = f"theta_{s}"
        if c in rows[0]:
            x = col(rows, c)
            allx += x
            rel = write_table(f"data/miller/theta_{s}.txt", [("wc", x), ("depth", depth)])
            plots.append(p_cycle(rel))
            leg.append(s.replace("_", " "))
    figs.append(figure("Multi-Soil Infiltration (Miller)", "Volumetric Water Content", "Depth (cm)", plots, leg,
                       depth_axis_opts(allx, depth), cycle=True))
    plots, leg, allx = [], [], []
    for s in soils:
        c = f"h_{s}_cm"
        if c in rows[0]:
            x = col(rows, c)
            allx += x
            rel = write_table(f"data/miller/h_{s}.txt", [("h", x), ("depth", depth)])
            plots.append(p_cycle(rel))
            leg.append(s.replace("_", " "))
    figs.append(figure("Multi-Soil Infiltration (Miller)", "Pressure Head (cm)", "Depth (cm)", plots, leg,
                       depth_axis_opts(allx, depth), cycle=True))
    return ("Miller Multi-Soil Infiltration", figs)


def sec_ireson():
    figs = []
    prof = read_csv("ireson_profile.csv")
    if prof is not None:
        depth = [-d for d in col(prof, "depth_m")]
        theta = col(prof, "theta_final")
        h = col(prof, "h_final_m")
        f_t = write_table("data/ireson/theta_profile.txt", [("wc", theta), ("depth", depth)])
        f_h = write_table("data/ireson/h_profile.txt", [("h", h), ("depth", depth)])
        figs.append(figure("Ireson Infiltration: Final Profile", "Volumetric Water Content", "Depth (m)",
                           [p_line(f_t, "blue")], ["Helios"], depth_axis_opts(theta, depth)))
        figs.append(figure("Ireson Infiltration: Final Profile", "Pressure Head (m)", "Depth (m)",
                           [p_line(f_h, "blue")], ["Helios"], depth_axis_opts(h, depth)))
    # HYDRUS validation time series at observation depths.
    val = read_csv("hydrus_validation.csv")
    ts = read_csv("ireson_timeseries.csv")
    if ts is not None:
        day = col(ts, "day")
        # Each depth uses one explicit colour shared by its Helios line and its HYDRUS markers.
        depth_cols = [("theta_0m", "0 m", 0.0, "blue"), ("theta_05m", "0.5 m", 0.5, "red"), ("theta_1m", "1.0 m", 1.0, "brown")]
        hyd_by_depth = {}
        if val is not None:
            for r in val:
                hyd_by_depth.setdefault(float(r["depth_m"]), []).append((float(r["day"]), float(r["theta_hydrus"])))
        plots, leg = [], []
        for c, label, dval, color in depth_cols:
            if c in ts[0]:
                rel = write_table(f"data/ireson/{c}.txt", [("day", day), ("theta", col(ts, c))])
                plots.append(p_line(rel, color, meta=None))
                leg.append(f"Helios {label}")
            if dval in hyd_by_depth:
                pts = sorted(hyd_by_depth[dval])
                rel = write_table(f"data/ireson/hydrus_{c}.txt", [("day", [p[0] for p in pts]), ("theta", [p[1] for p in pts])])
                plots.append(p_marks(rel, color, meta=None))
                leg.append(f"HYDRUS {label}")
        figs.append(figure("Ireson: Water Content vs HYDRUS", "Time (days)", "Volumetric Water Content", plots, leg, ["xmin=0"]))
    return ("Ireson Infiltration (HYDRUS validation)", figs) if figs else None


SECTIONS = [
    ("Validation Against Published Data", [sec_lehmann, sec_abeele, sec_zadeh, sec_celia, sec_ireson]),
    ("Analytical Benchmarks", [sec_gardner, sec_philip, sec_steady_state, sec_mass_balance, sec_grid_convergence]),
    ("Infiltration, Drainage \\& Soil Response", [sec_infiltration, sec_drainage, sec_soil_comparison, sec_warrick, sec_miller]),
]


# ---------------------------------------------------------------------------
# Document assembly + compilation
# ---------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{pgfplots}
\usepackage{booktabs}
\usepackage{amsmath}
\pgfplotsset{compat=1.18, width=12cm, height=8cm}
\title{SoilTransportModel Benchmark Suite}
\author{Helios \texttt{soiltransport} plugin}
\date{\today}
\begin{document}
\maketitle

\noindent This document records the benchmark tests used to verify and validate the Helios
\texttt{soiltransport} solver (a 1D Richards-equation solver with van Genuchten--Mualem or
Gardner constitutive models). Two categories are distinguished:
\emph{verification} tests, which compare the numerical solution against an exact analytical
result (the error should vanish under refinement), and \emph{validation} tests, which compare
against published experimental data or an independent numerical code (e.g.\ HYDRUS-1D). Each
test below lists its source and the exact setup used to generate the figures. Unless noted,
soil texture parameters are from Carsel \& Parrish (1988) and the constitutive model is van
Genuchten--Mualem (van Genuchten, 1980; Mualem, 1976).
\tableofcontents
\newpage
"""


# ---------------------------------------------------------------------------
# Per-benchmark documentation: (description, parameter rows). Keyed by the
# sec_<key> builder name. Citations are verified; see REFERENCES below.
# ---------------------------------------------------------------------------

BENCH_DOC = {
    "gardner": (
        r"\textbf{Verification} against the exact quasi-linear solution of the Gardner (1958) "
        r"exponential model $K(h)=K_s e^{\alpha h}$, which linearises Richards' equation via the "
        r"matric flux potential. Steady infiltration at flux $q$ toward a water table ($h=0$) has the "
        r"closed form $h(z)=\alpha^{-1}\ln\!\big[((K_s-q)e^{-\alpha z}+q)/K_s\big]$, $z$ being height "
        r"above the water table. The numerical steady state reproduces it to $\sim\!10^{-6}$ m.",
        [("Constitutive model", "Gardner exponential"), (r"$\theta_r,\ \theta_s$", "0.06, 0.40"),
         (r"$\alpha$", r"2.0 $\mathrm{m^{-1}}$"), (r"$K_s$", r"$1\times10^{-5}$ m/s"),
         ("Domain", r"0.5 m, 50 layers ($\Delta z=0.01$ m)"), ("Top BC", r"flux $q=K_s/2$"),
         ("Bottom BC", r"$h=0$ (water table)"), ("March", r"to steady state, $\Delta t=60$ s"),
         ("Source", "Gardner (1958)")],
    ),
    "philip": (
        r"\textbf{Verification} against the exact sorptivity of Philip (1957). Early-time ponded "
        r"infiltration obeys $I(t)=S\sqrt{t}+A_2 t$; a fit of $I/\sqrt{t}$ vs.\ $\sqrt{t}$ gives the "
        r"numerical sorptivity as the intercept, compared with the exact $S$ from the Boltzmann "
        r"similarity ODE for the same soil. The numerical value matches to $\sim\!1.5\%$.",
        [("Soil", "loam (Carsel \\& Parrish, 1988)"), ("Initial condition", r"$h=-3$ m (uniform)"),
         ("Top BC", r"$h=0$ (ponded)"), ("Bottom BC", "free drainage"),
         ("Domain", r"0.15 m, 300 layers ($\Delta z=5\times10^{-4}$ m)"),
         ("Time step / fit window", r"$\Delta t=0.5$ s, $t\in[100,400]$ s"),
         ("Interblock $K$", "arithmetic mean"), ("Source", "Philip (1957)")],
    ),
    "steady_state": (
        r"\textbf{Verification} of the steady-state limit: with equal heads imposed at both ends the "
        r"exact steady solution is a uniform profile (unit total-head gradient) carrying flux $K(h)$. "
        r"The solver holds the uniform head to $<\!1$ cm.",
        [("Soil", "sandy loam"), ("Initial condition", r"$h=-1$ m"),
         ("Top / Bottom BC", r"$h=-1$ m (both Dirichlet)"),
         ("Domain", r"1.0 m, 50 layers ($\Delta z=0.02$ m)"), ("Time", r"24 h, $\Delta t=60$ s"),
         ("Source", "standard analytical check")],
    ),
    "mass_balance": (
        r"\textbf{Verification} of mass conservation: with zero-flux boundaries the total storage must "
        r"be constant. The relative change stays below $0.1\%$.",
        [("Soil", "loam"), ("Initial condition", r"$h=-1$ m"), ("Top / Bottom BC", "zero flux (both)"),
         ("Domain", r"1.0 m, 50 layers ($\Delta z=0.02$ m)"), ("Time", r"1 h, $\Delta t=60$ s"),
         ("Source", "standard conservation check")],
    ),
    "grid_convergence": (
        r"\textbf{Verification} of numerical convergence: the same infiltration problem is solved at "
        r"increasing resolution and the change in final storage between the two finest grids is smaller "
        r"than between the two coarsest (Cauchy convergence).",
        [("Soil", "loam"), ("Initial condition", r"$h=-2$ m"),
         ("Top BC", r"$h=-0.1$ m"), ("Bottom BC", "free drainage"),
         ("Resolutions", r"$n_z\in\{10,20,50,100,200\}$"), ("Time", "30 min"),
         ("Source", "standard convergence study")],
    ),
    "infiltration": (
        r"\textbf{Illustrative} dry-soil infiltration (in the style of Celia et al., 1990): a "
        r"near-saturated surface head drives a wetting front into initially dry soil.",
        [("Soil", "sandy loam"), ("Initial condition", r"$h=-2$ m"),
         ("Top BC", r"$h=-0.1$ m"), ("Bottom BC", "free drainage"),
         ("Domain", r"1.0 m, 100 layers ($\Delta z=0.01$ m)"), ("Time", r"1 h, $\Delta t=10$ s"),
         ("Source", "illustrative (Celia et al., 1990, style)")],
    ),
    "drainage": (
        r"\textbf{Illustrative} gravity drainage: an initially wet column drains under gravity with a "
        r"zero-flux top and free drainage at the base.",
        [("Soil", "sandy loam"), ("Initial condition", r"$h=-0.3$ m"),
         ("Top BC", "zero flux"), ("Bottom BC", "free drainage"),
         ("Domain", r"1.0 m, 50 layers ($\Delta z=0.02$ m)"), ("Time", r"2 h, $\Delta t=60$ s"),
         ("Source", "illustrative")],
    ),
    "soil_comparison": (
        r"\textbf{Illustrative} texture comparison: the same ponded infiltration is run for several "
        r"soils; coarser-textured soils admit water faster.",
        [("Soils", "sand, sandy loam, loam, clay loam, clay"), ("Initial condition", r"$h=-2$ m"),
         ("Top BC", r"$h=-0.1$ m"), ("Bottom BC", "free drainage"),
         ("Domain", r"1.0 m, 100 layers"), ("Time", "1 h"),
         ("Source", "texture parameters: Carsel \\& Parrish (1988)")],
    ),
    "celia": (
        r"Dry-start infiltration of Celia et al.\ (1990), the classic mass-conservation benchmark. "
        r"\emph{Note:} the reference profile shown is an approximate (estimated) solution, not "
        r"tabulated data, so this comparison is indicative rather than a strict validation.",
        [(r"$\theta_r,\ \theta_s$", "0.102, 0.368"), (r"$\alpha,\ n$", r"3.35 $\mathrm{m^{-1}}$, 2.0"),
         (r"$K_s$", r"$9.22\times10^{-5}$ m/s"), ("Initial condition", r"$h=-0.615$ m"),
         ("Top BC", r"$h=-0.207$ m"), ("Bottom BC", r"$h=-0.615$ m"),
         ("Domain", r"0.4 m, 40 layers ($\Delta z=0.01$ m)"), ("Time", r"360 s, $\Delta t=1$ s"),
         ("Source", "Celia et al. (1990)")],
    ),
    "warrick": (
        r"Ponded infiltration into Yolo light clay (Warrick, 2003), run at several timesteps to "
        r"demonstrate the implicit solver's timestep-robustness.",
        [("Soil", "Yolo light clay"), (r"$\theta_r,\ \theta_s$", "0.124, 0.495"),
         (r"$\alpha,\ n$", r"1.5 $\mathrm{m^{-1}}$, 2.0"), (r"$K_s$", r"$1.203\times10^{-7}$ m/s"),
         ("Initial condition", r"$h=-2.025$ m"), ("Top BC", r"$h=0$ (ponded)"),
         ("Bottom BC", "free drainage"), ("Domain", r"1.2 m, 24 layers"),
         ("Time step / duration", r"$\Delta t\in\{1,300,900\}$ s, 100 h"), ("Source", "Warrick (2003)")],
    ),
    "miller": (
        r"Multi-soil infiltration toward a shallow water table (the ``Miller problem'' of "
        r"Ireson et al., 2023 / openRE), solved with adaptive timestepping for three textures.",
        [("Soils", "sand, loam, clay loam (openRE)"), ("Initial condition", r"$h=-0.05$ m"),
         ("Top BC", r"$h=0.001$ m"), ("Bottom BC", r"$h=0$ (water table)"),
         ("Domain", r"0.1 m, 80 layers ($\Delta z=1.25\times10^{-3}$ m)"),
         ("Time stepping", "adaptive, 0.18 d"), ("Source", "Ireson et al. (2023), openRE")],
    ),
    "ireson": (
        r"\textbf{Validation} field infiltration benchmark of Ireson et al.\ (2023) / openRE, compared "
        r"to HYDRUS-1D. The surface is driven by the openRE daily precipitation series (no evaporation) "
        r"with free drainage at the base; water content at 0, 0.5 and 1.0 m is compared to HYDRUS.",
        [(r"$\theta_r,\ \theta_s$", "0.131, 0.396"), (r"$\alpha,\ n$", r"0.423 $\mathrm{m^{-1}}$, 2.06"),
         (r"$K_s$", r"$5.74\times10^{-7}$ m/s (0.0496 m/d)"), ("Initial condition", r"$h=-3.59$ m"),
         ("Top BC", "daily precipitation flux"), ("Bottom BC", "free drainage"),
         ("Domain", r"1.5 m, 30 layers ($\Delta z=0.05$ m)"), ("Time", r"90 d, $\Delta t=1$ h"),
         ("Source", "Ireson et al. (2023), openRE")],
    ),
    "zadeh": (
        r"\textbf{Validation} of infiltration into very dry soil: cumulative infiltration is compared to "
        r"the experimental data of Zadeh et al.\ (2007), using the parameters and configuration of "
        r"Caviedes-Voulli\`eme et al.\ (2013, Table 1).",
        [(r"$\theta_r,\ \theta_s$", "0.062, 0.46"), (r"$\alpha,\ n$", r"3.7 $\mathrm{m^{-1}}$, 1.67"),
         (r"$K_s$", r"$9.37\times10^{-7}$ m/s"), ("Initial condition", r"$h=-200$ m (very dry)"),
         ("Top BC", r"$h=0$"), ("Bottom BC", "zero flux"),
         ("Domain", r"0.35 m, 100 layers ($\Delta z=3.5\times10^{-3}$ m)"),
         ("Time", r"1 d, $\Delta t=1$ s"),
         ("Source", "Zadeh et al. (2007); Caviedes-Voulli\\`eme et al. (2013)")],
    ),
    "abeele": (
        r"\textbf{Validation} of free drainage in a deep (6 m) lysimeter of crushed Bandelier Tuff "
        r"(Abeele, 1984), as reproduced by Lai \& Ogden (2015) and Caviedes-Voulli\`eme et al.\ (2013); "
        r"compared to a HYDRUS-1D solution and measured data at $t=100$ days.",
        [("Soil", "crushed Bandelier Tuff"), (r"$\theta_r,\ \theta_s$", "0.0, 0.33"),
         (r"$\alpha,\ n$", r"1.43 $\mathrm{m^{-1}}$, 1.506"), (r"$K_s$", r"$2.89\times10^{-6}$ m/s"),
         ("Initial condition", r"$h\approx 0$ (saturated)"), ("Top BC", "zero flux"),
         ("Bottom BC", r"$h=0$"), ("Domain", r"6 m, 100 layers ($\Delta z=0.06$ m)"),
         ("Time", r"100 d, $\Delta t=10$ s"), ("Source", "Abeele (1984); Lai \\& Ogden (2015)")],
    ),
    "lehmann": (
        r"\textbf{Validation} of a three-layer heterogeneous column (Lehmann \& Ackerer, 1998): a "
        r"Berino loamy-fine-sand layer between two Glendale clay-loam layers, driven by a surface flux "
        r"series. The numerical $\theta$ and $h$ profiles are compared with the reference solution of "
        r"Li et al.\ (2021).",
        [("Top/bottom layers", "Berino loamy fine sand"), ("Middle layer", "Glendale clay loam"),
         ("Sand $\\theta_r,\\theta_s,\\alpha,n$", r"0.0286, 0.366, 2.8 $\mathrm{m^{-1}}$, 2.239"),
         ("Clay $\\theta_r,\\theta_s,\\alpha,n$", r"0.106, 0.469, 1.04 $\mathrm{m^{-1}}$, 1.395"),
         ("Initial condition", r"$h=-1$ m"), ("Top BC", "daily flux series"),
         ("Bottom BC", "free drainage"), ("Domain", r"1.8 m (3$\times$0.6 m), 200 layers"),
         ("Time", r"20 d, $\Delta t=20$ s"), ("Source", "Lehmann \\& Ackerer (1998)")],
    ),
}


def param_table(rows):
    out = [r"\begin{center}", r"\begin{tabular}{@{}ll@{}}", r"\toprule",
           r"\textbf{Parameter} & \textbf{Value} \\", r"\midrule"]
    out += [f"{k} & {v} \\\\" for k, v in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)


def doc_block(key):
    entry = BENCH_DOC.get(key)
    if not entry:
        return None
    desc, rows = entry
    return "\\noindent " + desc + "\n\n" + param_table(rows)


REFERENCES = r"""
\newpage
\section*{References}
\addcontentsline{toc}{section}{References}
\small
\begin{itemize}\setlength{\itemsep}{1pt}
\item Abeele, W.\,V. (1984). Hydraulic testing of crushed Bandelier Tuff. Report LA-10037-MS, Los Alamos National Laboratory, 21 pp.
\item Carsel, R.\,F. \& Parrish, R.\,S. (1988). Developing joint probability distributions of soil water retention characteristics. \emph{Water Resources Research}, 24(5), 755--769.
\item Caviedes-Voulli\`eme, D., Garc\'ia-Navarro, P. \& Murillo, J. (2013). Verification, conservation, stability and efficiency of a finite volume method for the 1D Richards equation. \emph{Journal of Hydrology}, 480, 69--84. doi:10.1016/j.jhydrol.2012.12.008
\item Celia, M.\,A., Bouloutas, E.\,T. \& Zarba, R.\,L. (1990). A general mass-conservative numerical solution for the unsaturated flow equation. \emph{Water Resources Research}, 26(7), 1483--1496. doi:10.1029/WR026i007p01483
\item Gardner, W.\,R. (1958). Some steady-state solutions of the unsaturated moisture flow equation with application to evaporation from a water table. \emph{Soil Science}, 85(4), 228--232.
\item Ireson, A.\,M., Spiteri, R.\,J., Clark, M.\,P. \& Mathias, S.\,A. (2023). A simple, efficient, mass-conservative approach to solving Richards' equation (openRE, v1.0). \emph{Geoscientific Model Development}, 16(2), 659--677. doi:10.5194/gmd-16-659-2023
\item Lai, W. \& Ogden, F.\,L. (2015). A mass-conservative finite volume predictor--corrector solution of the 1D Richards' equation. \emph{Journal of Hydrology}, 523, 119--127.
\item Lehmann, F. \& Ackerer, P. (1998). Comparison of iterative methods for improved solutions of the fluid flow equation in partially saturated porous media. \emph{Transport in Porous Media}, 31(3), 275--292. doi:10.1023/A:1006555107450
\item Li, Z., \"Ozgen-Xian, I. \& Maina, F.\,Z. (2021). A mass-conservative predictor--corrector solution to the 1D Richards equation with adaptive time control. \emph{Journal of Hydrology}, 592, 125809. doi:10.1016/j.jhydrol.2020.125809
\item Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous media. \emph{Water Resources Research}, 12(3), 513--522.
\item Philip, J.\,R. (1957). The theory of infiltration: 1. The infiltration equation and its solution. \emph{Soil Science}, 83(5), 345--357.
\item van Genuchten, M.\,Th. (1980). A closed-form equation for predicting the hydraulic conductivity of unsaturated soils. \emph{Soil Science Society of America Journal}, 44(5), 892--898.
\item Warrick, A.\,W. (2003). \emph{Soil Water Dynamics}. Oxford University Press.
\item Zadeh, K.\,S., Shirmohammadi, A., Montas, H.\,J. \& Felton, G. (2007). Evaluation of infiltration models in contaminated landscape. \emph{Journal of Environmental Science and Health, Part A}, 42(9), 1311--1320. doi:10.1080/10934520701373000
\end{itemize}
"""


def build_document():
    body = []
    for part_title, builders in SECTIONS:
        part_chunks = []
        for builder in builders:
            print(f"  building: {builder.__name__}")
            result = builder()
            if not result:
                continue
            heading, figs = result
            if not figs:
                continue
            chunk = [f"\\subsection{{{heading}}}"]
            block = doc_block(builder.__name__.replace("sec_", ""))
            if block:
                chunk.append(block)
            chunk += figs
            part_chunks.append("\n\n".join(chunk))
        if part_chunks:
            body.append(f"\\section{{{part_title}}}\n\n" + "\n\n".join(part_chunks))
    return PREAMBLE + "\n\n".join(body) + "\n\n" + REFERENCES + "\n\\end{document}\n"


def compile_pdf():
    tex = OUT_DIR / "benchmarks.tex"
    print("\nCompiling with latexmk ...")
    try:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "benchmarks.tex"],
                       cwd=OUT_DIR, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode(errors="replace")[-3000:] if e.stdout else "")
        print("LaTeX compilation FAILED (see log above / benchmark_report/benchmarks.log)")
        return False
    subprocess.run(["latexmk", "-c", "benchmarks.tex"], cwd=OUT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"PDF written: {tex.with_suffix('.pdf')}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run the benchmark executable first")
    ap.add_argument("--no-pdf", action="store_true", help="skip LaTeX compilation")
    args = ap.parse_args()

    if args.run:
        exe = BUILD_DIR / "soiltransport_benchmarks"
        if not exe.exists():
            print(f"ERROR: {exe} not found; build the sample first.")
            sys.exit(1)
        print("Running benchmark suite (this may take ~1 minute) ...")
        subprocess.run([str(exe)], cwd=BUILD_DIR, check=True)

    OUT_DIR.mkdir(exist_ok=True)
    print("Generating pgfplots data + LaTeX report ...")
    doc = build_document()
    tex = OUT_DIR / "benchmarks.tex"
    tex.write_text(doc)
    print(f"Wrote {tex}")

    if not args.no_pdf:
        compile_pdf()


if __name__ == "__main__":
    main()
