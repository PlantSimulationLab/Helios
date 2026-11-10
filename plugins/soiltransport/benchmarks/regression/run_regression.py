#!/usr/bin/env python3
"""SoilTransport regression gate: build the benchmark cases against a Helios tree, run them, and compare the results
with the baseline (the last accepted version of the plug-in) so that a change cannot silently make the solvers less
accurate, more work or slower.

Commands
  build <helios_dir> [--build-dir DIR]          build soil_regression against a Helios source tree
  run --exe EXE --label NAME [--tier quick|full] [--repeat N] [--out FILE]
                                                 run the cases and write a results file
  gate --exe EXE [--tier quick|full] [--baseline FILE]
                                                 run, then compare with the baseline (the newest file in baselines/
                                                 unless given); exit 1 on a regression. The check after every change.
  compare CURRENT BASELINE [BASELINE ...]        compare a results file with one or more baselines
  accept RESULTS                                 make a results file the new baseline (copied into baselines/)
  history [FILE ...]                             table of the main metrics across baselines (default: baselines/)

Keep the baseline fixed between deliberate acceptances (a commit, or an intended trade-off): then small changes
that each pass the tolerances still add up against the same reference and eventually fail the gate.

What counts as a regression (against a baseline in which the case passed)
  status   the case now fails, times out or crashes
  errors   an error against an exact or reference solution grows by more than --err-tol (5%) and 1e-7
  values   a result without an exact reference changes by more than --value-tol (0.1%)
  work     steps or iterations grow by more than --work-tol (10%) and 1
  time     the solve time grows by more than --time-tol (15%) and --time-floor (2 ms); only against baselines on the
           same machine (CPU and GPU), since times are not comparable across machines
A deliberate trade-off is accepted only through waivers.json: {"case", "metric", "baseline", "reason"} entries
(baseline and metric may be "*").
"""
import argparse
import datetime
import glob
import json
import os
import platform
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(HERE, "baselines")


def machine():
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return {"host": socket.gethostname(), "cpu": cpu, "gpu": gpu}


def cmd_build(args):
    build = os.path.abspath(args.build_dir or os.path.join(HERE, "build", os.path.basename(os.path.abspath(args.helios_dir))))
    os.makedirs(build, exist_ok=True)
    subprocess.run(["cmake", HERE, "-DCMAKE_BUILD_TYPE=Release", "-DHELIOS_DIR=" + os.path.abspath(args.helios_dir)], cwd=build, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", ".", "--parallel", str(os.cpu_count() or 4)], cwd=build, check=True, stdout=subprocess.DEVNULL)
    exe = os.path.join(build, "soil_regression")
    print(exe)
    return exe


def run_case(exe, name, timeout):
    env = dict(os.environ)
    env.setdefault("SOIL_REGRESSION_REFERENCE", os.path.join(HERE, "reference"))
    try:
        p = subprocess.run([exe, "run", name], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"case": name, "status": "timeout", "note": "over %g s" % timeout, "time_s": timeout, "errors": {}, "values": {}, "work": {}}
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or [""]
        return {"case": name, "status": "crashed", "note": "exit %d: %s" % (p.returncode, tail[0][:200]), "time_s": 0.0, "errors": {}, "values": {}, "work": {}}
    return json.loads(lines[-1])


def same_metrics(a, b, rel=1e-6):
    """Repeat runs must agree (to round-off: threaded sums may add in a different order)"""
    for g in ("errors", "values", "work"):
        if a[g].keys() != b[g].keys():
            return False
        for k, v in a[g].items():
            if abs(v - b[g][k]) > rel * max(abs(v), abs(b[g][k])) + 1e-14: # round-off-sized quantities vary more
                return False
    return True


def cmd_run(args):
    exe = os.path.abspath(args.exe)
    names = [l.split()[0] for l in subprocess.run([exe, "list", args.tier], capture_output=True, text=True, check=True).stdout.splitlines() if l.strip()]
    if args.cases:
        names = [n for n in names if any(n.startswith(c) for c in args.cases)]
    results = {}
    for name in names:
        r = run_case(exe, name, args.timeout)
        # Repeat quick cases and keep the fastest run (results must be deterministic; time is noisy)
        n = 1
        while r["status"] == "ok" and n < args.repeat and r["time_s"] < args.repeat_below:
            again = run_case(exe, name, args.timeout)
            n += 1
            if again["status"] != "ok":
                r = again
                break
            if not same_metrics(again, r):
                r["note"] = "nondeterministic"
            r["time_s"] = min(r["time_s"], again["time_s"])
        results[name] = r
        print("  %-30s %-11s %8.3f s  %s" % (name, r["status"], r["time_s"], r.get("note", "")), flush=True)
    out = {"label": args.label, "date": datetime.datetime.now().isoformat(timespec="seconds"), "machine": machine(), "exe": os.path.basename(exe), "tier": args.tier, "repeat": args.repeat, "cases": results}
    path = args.out or os.path.join(HERE, "results", args.label + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("results written to", path)
    return path


def load(path):
    with open(path) as f:
        return json.load(f)


def load_waivers(path):
    if path and os.path.exists(path):
        return load(path)
    return []


def waived(waivers, case, metric, baseline):
    for w in waivers:
        if w.get("case") in (case, "*") and w.get("metric") in (metric, "*") and w.get("baseline") in (baseline, "*"):
            return w.get("reason", "waived")
    return None


def same_machine(a, b):
    return a["machine"].get("cpu") == b["machine"].get("cpu") and a["machine"].get("gpu") == b["machine"].get("gpu")


def compare(current, baselines, args):
    """Returns (regressions, improvements, notes): lists of strings"""
    waivers = load_waivers(args.waivers)
    regressions, improvements, notes = [], [], []
    for base in baselines:
        bl = base["label"]
        timed = same_machine(current, base)
        if not timed:
            notes.append("%s: recorded on another machine (%s / %s): times not compared" % (bl, base["machine"].get("cpu"), base["machine"].get("gpu")))
        for name, b in base["cases"].items():
            c = current["cases"].get(name)
            if c is None or b["status"] == "unsupported":
                continue
            if b["status"] != "ok":
                if c["status"] == "ok":
                    improvements.append("%-28s %-30s %s -> ok  [vs %s]" % (name, "status", b["status"], bl))
                continue
            def flag(metric, msg, worse=True):
                reason = waived(waivers, name, metric, bl) if worse else None
                line = "%-28s %-30s %s  [vs %s]" % (name, metric, msg, bl)
                if not worse:
                    improvements.append(line)
                elif reason:
                    notes.append(line + "  WAIVED: " + reason)
                else:
                    regressions.append(line)
            if c["status"] != "ok":
                flag("status", "%s (%s)" % (c["status"], c.get("note", "")))
                continue
            for m, bv in b["errors"].items():
                cv = c["errors"].get(m)
                if cv is None:
                    continue
                if cv > bv * (1 + args.err_tol) and cv - bv > 1e-7:
                    flag(m, "%.4g -> %.4g" % (bv, cv))
                elif cv < bv * (1 - args.err_tol) and bv - cv > 1e-7:
                    flag(m, "%.4g -> %.4g" % (bv, cv), worse=False)
            for m, bv in b["values"].items():
                cv = c["values"].get(m)
                if cv is not None and abs(cv - bv) > args.value_tol * max(abs(bv), 1e-300):
                    flag(m, "changed %.6g -> %.6g" % (bv, cv))
            for m, bv in b["work"].items():
                cv = c["work"].get(m)
                if cv is None:
                    continue
                if cv > bv * (1 + args.work_tol) + 1:
                    flag(m, "%.6g -> %.6g" % (bv, cv))
                elif cv < bv * (1 - args.work_tol) - 1:
                    flag(m, "%.6g -> %.6g" % (bv, cv), worse=False)
            if timed:
                bt, ct = b["time_s"], c["time_s"]
                if ct > bt * (1 + args.time_tol) + args.time_floor:
                    flag("time_s", "%.3f -> %.3f s (%+.0f%%)" % (bt, ct, 100 * (ct / bt - 1)))
                elif ct < bt * (1 - args.time_tol) - args.time_floor:
                    flag("time_s", "%.3f -> %.3f s (%+.0f%%)" % (bt, ct, 100 * (ct / bt - 1)), worse=False)
    return regressions, improvements, notes


def report(current, baselines, args):
    regressions, improvements, notes = compare(current, baselines, args)
    print("\nCompared %s with %s" % (current["label"], ", ".join(b["label"] for b in baselines)))
    for title, lines in (("Notes", notes), ("Improvements", improvements), ("REGRESSIONS", regressions)):
        if lines:
            print("\n%s (%d):" % (title, len(lines)))
            for l in lines:
                print("  " + l)
    print("\n%s" % ("FAIL: %d regression(s)" % len(regressions) if regressions else "PASS: no regressions"))
    return 1 if regressions else 0


def baseline_files():
    files = sorted(glob.glob(os.path.join(BASELINE_DIR, "*.json")))
    return files


def cmd_compare(args):
    return report(load(args.current), [load(p) for p in args.baselines], args)


def latest_baseline():
    files = baseline_files()
    return max(files, key=lambda p: load(p)["date"]) if files else None


def cmd_gate(args):
    args.label = args.label or "current"
    base = args.baseline or latest_baseline()
    if not base:
        print("no baseline in", BASELINE_DIR, "- record one with run + accept")
        return 1
    path = cmd_run(args)
    return report(load(path), [load(base)], args)


def cmd_accept(args):
    r = load(args.results)
    dst = os.path.join(BASELINE_DIR, r["label"] + ".json")
    os.makedirs(BASELINE_DIR, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(r, f, indent=1)
    print("baseline", dst)
    return 0


def primary(r):
    """The case's first error metric (its main accuracy measure)"""
    if r["status"] != "ok":
        return r["status"]
    if r["errors"]:
        k, v = next(iter(r["errors"].items()))
        return "%.3g" % v
    if r["values"]:
        k, v = next(iter(r["values"].items()))
        return "%.6g" % v
    return "ok"


def cmd_history(args):
    runs = [load(p) for p in (args.files or baseline_files())]
    names = []
    for r in runs:
        for n in r["cases"]:
            if n not in names:
                names.append(n)
    w = 18
    print("%-30s" % "case (main error / solve time)" + "".join("%*s" % (w, r["label"][:w - 1]) for r in runs))
    for n in names:
        row1, row2 = "%-30s" % n, "%-30s" % ""
        for r in runs:
            c = r["cases"].get(n)
            if c is None:
                row1 += "%*s" % (w, "-")
                row2 += "%*s" % (w, "")
                continue
            row1 += "%*s" % (w, primary(c) if c["status"] != "unsupported" else "n/a")
            row2 += "%*s" % (w, ("%.3f s" % c["time_s"]) if c["status"] == "ok" else "")
        print(row1)
        print(row2)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("helios_dir")
    b.add_argument("--build-dir")
    for name in ("run", "gate"):
        r = sub.add_parser(name)
        r.add_argument("--exe", required=True)
        r.add_argument("--label", required=(name == "run"))
        r.add_argument("--tier", default="quick", choices=["quick", "full"])
        r.add_argument("--repeat", type=int, default=3, help="runs of each quick case; the fastest time is kept")
        r.add_argument("--repeat-below", type=float, default=20.0, help="only repeat cases faster than this [s]")
        r.add_argument("--timeout", type=float, default=600.0)
        r.add_argument("--cases", nargs="*", help="only cases starting with these names")
        r.add_argument("--out")
        if name == "gate":
            r.add_argument("--baseline", help="baseline results file (default: the newest in baselines/)")
    a = sub.add_parser("accept")
    a.add_argument("results")
    c = sub.add_parser("compare")
    c.add_argument("current")
    c.add_argument("baselines", nargs="+")
    h = sub.add_parser("history")
    h.add_argument("files", nargs="*")
    for p in (sub.choices["gate"], c):
        p.add_argument("--err-tol", type=float, default=0.05)
        p.add_argument("--value-tol", type=float, default=1e-3)
        p.add_argument("--work-tol", type=float, default=0.10)
        p.add_argument("--time-tol", type=float, default=0.15)
        p.add_argument("--time-floor", type=float, default=0.002, help="absolute time change ignored [s]")
        p.add_argument("--waivers", default=os.path.join(HERE, "waivers.json"))
    args = ap.parse_args()
    return {"build": lambda a: (cmd_build(a), 0)[1], "run": lambda a: (cmd_run(a), 0)[1], "compare": cmd_compare, "gate": cmd_gate, "accept": cmd_accept, "history": cmd_history}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
