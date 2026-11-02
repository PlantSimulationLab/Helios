"""Compare link trajectories from physicsengine_isaac_comparison (MuJoCo) and isaac_box_sweep.py (PhysX).

Both write the same CSV: time, box position, contact force on the plant, the peak contact force and penetration since
the previous row, the contact count, and then x, y, z of every link in the order of <plant>_links.csv.

Usage (numpy and matplotlib are needed; IsaacSim's Python has both):
    python compare.py --links out/apple_links.csv --sweep out/apple_sweep.json --run "MuJoCo 2 ms=out/mujoco_apple_dt0.002_noarmature.csv" --run "PhysX 2 ms=out/isaac_apple_dt0.002_gpu.csv" --out results/apple
"""

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--links", required=True)
parser.add_argument("--sweep", required=True)
parser.add_argument("--run", action="append", required=True, help='"label=path.csv"; the first run is the reference for link-by-link differences')
parser.add_argument("--out", required=True, help="output prefix for <prefix>_summary.json, <prefix>_summary.md and plots")
parser.add_argument("--plots", action="store_true")
args = parser.parse_args()

LINK_TYPES = ["internode", "petiole", "peduncle", "leaf", "fruit", "flower"]

link_rows = [line.split(",") for line in open(args.links).read().splitlines()]
column = {name: i for i, name in enumerate(link_rows[0])}
link_rows = link_rows[1:]
link_type = np.array([LINK_TYPES[int(r[column["type"]])] for r in link_rows])
branch_order = np.array([int(r[column["branch_order"]]) for r in link_rows])
rest = np.array([[float(r[column[f"rest_{a}"]]) for a in "xyz"] for r in link_rows])
sweep = json.load(open(args.sweep))
settle_time = sweep["settle_time"]
height = sweep["height"]
height_bin = np.clip((rest[:, 2] / height * 4).astype(int), 0, 3)  # quarters of the tree height
HEIGHT_LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]


def load(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {
        "time": data[:, 0],
        "box": data[:, 1:4],
        "force": data[:, 4:7],
        "peak_force": data[:, 7],
        "peak_penetration": data[:, 8],
        "contacts": data[:, 9],
        "positions": data[:, 10:].reshape(len(data), -1, 3),
    }


def at_time(run, t):
    """Link positions at time t, interpolated linearly between rows, so runs with different time steps line up."""
    times = run["time"]
    k = int(np.searchsorted(times, t))
    if k <= 0:
        return run["positions"][0]
    if k >= len(times):
        return run["positions"][-1]
    w = (t - times[k - 1]) / (times[k] - times[k - 1])
    return (1 - w) * run["positions"][k - 1] + w * run["positions"][k]


def grouped_median(values, groups, labels=None):
    result = {}
    for g in np.unique(groups):
        key = labels[g] if labels is not None else str(g)
        result[key] = float(np.median(values[groups == g]))
    return result


runs = []
for spec in args.run:
    label, path = spec.split("=", 1)
    run = load(path)
    run["label"] = label
    run["path"] = path
    summary_path = path.replace(".csv", ".json")
    run["meta"] = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    runs.append(run)

summary = {"plant": sweep["plant"], "height_m": height, "links": len(rest), "runs": {}}
reference = runs[0]
reference_settled = at_time(reference, settle_time)
reference_sweep_peak = None

for run in runs:
    finished = run["time"][-1] >= sweep["total_time"] - 0.05
    settled = at_time(run, settle_time)
    sag = np.linalg.norm(settled - rest, axis=1)
    sweep_rows = run["time"] > settle_time
    # Largest displacement of each link during the sweep, from where it had settled, and from the exported rest pose
    from_settled = np.linalg.norm(run["positions"][sweep_rows] - settled[None], axis=2).max(axis=0) if sweep_rows.any() else np.zeros(len(rest))
    from_rest = np.linalg.norm(run["positions"] - rest[None], axis=2).max(axis=0)
    # Motion in the last 0.1 s of settling shows whether the plant had come to rest
    late = (run["time"] > settle_time - 0.1) & (run["time"] <= settle_time)
    settle_motion = float(np.linalg.norm(run["positions"][late] - settled[None], axis=2).max()) if late.any() else float("nan")
    contact_rows = run["contacts"] > 0
    entry = {
        "finished": bool(finished),
        "simulated_until_s": float(run["time"][-1]),
        "sag_after_settling_m": {
            "median": float(np.median(sag)),
            "p90": float(np.percentile(sag, 90)),
            "max": float(sag.max()),
            "by_type": grouped_median(sag, link_type),
            "by_branch_order": grouped_median(sag, branch_order),
            "by_height": grouped_median(sag, height_bin, HEIGHT_LABELS),
        },
        "still_moving_at_end_of_settling_m": settle_motion,
        "sweep_displacement_from_settled_m": {
            "median": float(np.median(from_settled)),
            "p90": float(np.percentile(from_settled, 90)),
            "max": float(from_settled.max()),
            "links_moved_over_1cm": int(np.sum(from_settled > 0.01)),
            "by_type": grouped_median(from_settled, link_type),
            "by_branch_order": grouped_median(from_settled, branch_order),
            "by_height": grouped_median(from_settled, height_bin, HEIGHT_LABELS),
            "by_height_p90": {HEIGHT_LABELS[g]: float(np.percentile(from_settled[height_bin == g], 90)) for g in np.unique(height_bin)},
        },
        "max_displacement_from_rest_m": float(from_rest.max()),
        "contact": {
            "rows_with_contact": int(contact_rows.sum()),
            "peak_force_N": float(run["peak_force"].max()),
            "median_peak_force_in_contact_N": float(np.median(run["peak_force"][contact_rows])) if contact_rows.any() else 0.0,
            "peak_penetration_mm": float(1000 * run["peak_penetration"].max()),
            "median_peak_penetration_in_contact_mm": float(1000 * np.median(run["peak_penetration"][contact_rows])) if contact_rows.any() else 0.0,
            "max_contacts": int(run["contacts"].max()),
        },
        "meta": run["meta"],
    }
    if run is not reference:
        # Link-by-link differences from the reference run
        sag_reference = np.linalg.norm(reference_settled - rest, axis=1)
        settled_difference = np.linalg.norm(settled - reference_settled, axis=1)
        reference_from_settled = summary["runs"][reference["label"]]["_from_settled"]
        entry["vs_reference"] = {
            "settled_position_difference_m": {"median": float(np.median(settled_difference)), "p90": float(np.percentile(settled_difference, 90)), "max": float(settled_difference.max())},
            "sag_ratio_median": float(np.median(sag[sag_reference > 1e-3] / sag_reference[sag_reference > 1e-3])) if np.any(sag_reference > 1e-3) else float("nan"),
            "sag_correlation": float(np.corrcoef(sag, sag_reference)[0, 1]),
            "sweep_displacement_correlation": float(np.corrcoef(from_settled, reference_from_settled)[0, 1]) if sweep_rows.any() else float("nan"),
        }
    entry["_from_settled"] = from_settled
    entry["_sag"] = sag
    summary["runs"][run["label"]] = entry

# Plots
if args.plots:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for run in runs:
        displacement = np.linalg.norm(run["positions"] - rest[None], axis=2)
        axes[0].plot(run["time"], 1000 * np.median(displacement, axis=1), label=run["label"])
        axes[1].plot(run["time"], 1000 * displacement.max(axis=1), label=run["label"])
        axes[2].plot(run["time"], run["peak_force"], label=run["label"])
    axes[0].set_ylabel("median link displacement from rest pose (mm)")
    axes[1].set_ylabel("largest link displacement from rest pose (mm)")
    axes[2].set_ylabel("peak contact force in each sample (N)")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.axvline(settle_time, color="0.6", lw=0.8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out + "_time.png", dpi=110)

    if len(runs) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        ref = summary["runs"][reference["label"]]
        for run in runs[1:]:
            entry = summary["runs"][run["label"]]
            axes[0].scatter(1000 * ref["_sag"], 1000 * entry["_sag"], s=4, alpha=0.5, label=run["label"])
            axes[1].scatter(1000 * ref["_from_settled"], 1000 * entry["_from_settled"], s=4, alpha=0.5, label=run["label"])
        for ax, title in zip(axes, ["sag after settling, per link (mm)", "largest displacement during sweep, per link (mm)"]):
            limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, limit], [0, limit], color="0.5", lw=0.8)
            ax.set_xlabel(reference["label"])
            ax.set_ylabel("other run")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out + "_links.png", dpi=110)

for entry in summary["runs"].values():
    del entry["_from_settled"]
    del entry["_sag"]

with open(args.out + "_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# A short table for the report
lines = [
    "| Run | Finished | Sag median / max (mm) | Still moving at end of settling (mm) | Sweep displacement median / p90 / max (mm) | Peak force (N) | Peak penetration (mm) |",
    "|---|---|---|---|---|---|---|",
]
for label, entry in summary["runs"].items():
    sag = entry["sag_after_settling_m"]
    moved = entry["sweep_displacement_from_settled_m"]
    contact = entry["contact"]
    lines.append(
        f"| {label} | {'yes' if entry['finished'] else 'no, ' + format(entry['simulated_until_s'], '.2f') + ' s'} | {1000 * sag['median']:.1f} / {1000 * sag['max']:.0f} | {1000 * entry['still_moving_at_end_of_settling_m']:.1f} | "
        f"{1000 * moved['median']:.1f} / {1000 * moved['p90']:.0f} / {1000 * moved['max']:.0f} | {contact['peak_force_N']:.1f} | {contact['peak_penetration_mm']:.1f} |"
    )
open(args.out + "_summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
