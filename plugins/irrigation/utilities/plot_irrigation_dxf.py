#!/usr/bin/env python3
"""
plot_irrigation_dxf.py
----------------------

Minimal DXF reader / viewer for Helios irrigation plug-ins.

▪  Reads ASCII DXF files (LINE, LWPOLYLINE, POLYLINE/VERTEX/SEQEND).
▪  Merges coincident vertices within --tol (default 1 µm).
▪  Skips any entity on --ignore-layers (default: layer “0”).
▪  Optional extra filters
   – --min-length L        → discard segments shorter than L m
   – --skip-closed         → discard closed polylines (symbols)
   – --orth-only           → keep only horizontal/vertical segments
▪  Re-bases coords so min(x)=min(y)=0 if --rebase is given.
▪  Quick matplotlib plot unless --no-plot.

© 2025 Brian Bailey
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
from collections import Counter
from typing   import List, Tuple

# ───────────────────────────────── argparse ───────────────────────────────
def get_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('dxf', help='ASCII DXF file')
    ap.add_argument('--ignore-layers', default='0',
                    help='comma-separated layer names to skip')
    ap.add_argument('--min-length', type=float, default=0.0, metavar='L',
                    help='discard segments shorter than L m')
    ap.add_argument('--skip-closed', action='store_true',
                    help='discard closed polylines')
    ap.add_argument('--orth-only', action='store_true',
                    help='keep only horizontal / vertical segments')
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='node-merge tolerance (m)')
    ap.add_argument('--rebase', action='store_true',
                    help='shift so min(x)=min(y)=0')
    ap.add_argument('--no-plot', action='store_true',
                    help="don't launch matplotlib")
    ap.add_argument('--list-layers', action='store_true',
                    help='print every layer name in the file and exit')
    ap.add_argument('--show-pressure', action='store_true',
                    help='colour nodes by TEXT strings on layer PRESSURE')
    return ap.parse_args()

# ───────────────────────────── small utilities ────────────────────────────
def read_dxf_lines(path: Path) -> List[str]:
    try:
        with path.open('rt', encoding='utf-8', errors='ignore') as f:
            return [ln.strip() for ln in f]
    except FileNotFoundError:
        sys.exit(f'ERROR: cannot open “{path}”')

def collect_pairs(lines: List[str], i: int) -> tuple[dict[int,str], int]:
    """Return ({code: value,…}, new_index) until next “0” or EOF."""
    m: dict[int,str] = {}
    n = len(lines)
    while i + 1 < n and lines[i] != '0':
        m[int(lines[i])] = lines[i+1]
        i += 2
    return m, i

class NodeBank:
    """Merge coincident nodes within tol."""
    def __init__(self, tol: float):
        self.tol2 = tol * tol
        self.xy   : List[Tuple[float,float]] = []

    def index(self, x: float, y: float) -> int:
        for k,(px,py) in enumerate(self.xy):
            if (px-x)**2 + (py-y)**2 < self.tol2:
                return k
        self.xy.append((x,y))
        return len(self.xy) - 1

def keep_seg(x1,y1,x2,y2, *, tol, orth_only, min_len) -> bool:
    if orth_only:
        dx,dy = abs(x2-x1), abs(y2-y1)
        if dx >= tol and dy >= tol:          # diagonal → drop
            return False
    return math.hypot(x2-x1, y2-y1) >= min_len

# ─────────────────────────────── DXF parser ───────────────────────────────
def parse_dxf(lines: List[str], *,
              ignore_layers: set[str],
              tol: float,
              min_len: float,
              skip_closed: bool,
              orth_only: bool
              ) -> tuple[List[Tuple[float,float]],
List[Tuple[int,int]],
Counter]:
    """Return (nodes, pipes, layer_count)."""
    nodes = NodeBank(tol)
    pipes : List[Tuple[int,int]] = []
    layer_seen = Counter()

    i,n = 0,len(lines)
    while i + 1 < n:
        if lines[i] != '0':
            i += 1; continue
        tag = lines[i+1]; i += 2

        # --- LINE ---
        if tag == 'LINE':
            ent,i = collect_pairs(lines,i)
            lyr   = ent.get(8, '')
            layer_seen[lyr] += 1
            if lyr in ignore_layers: continue
            if not all(k in ent for k in (10,20,11,21)): continue
            x1,y1,x2,y2 = map(float,(ent[10],ent[20],ent[11],ent[21]))
            if not keep_seg(x1,y1,x2,y2,tol=tol,orth_only=orth_only,min_len=min_len):
                continue
            pipes.append((nodes.index(x1,y1), nodes.index(x2,y2)))

        # --- LWPOLYLINE ---
        elif tag == 'LWPOLYLINE':
            verts: List[Tuple[float,float]] = []
            cur_x  : float|None = None
            closed = False
            lyr = ''
            while i + 1 < n and lines[i] != '0':
                code,val = int(lines[i]), lines[i+1]
                if code == 8: lyr = val
                if code == 10: cur_x = float(val)
                elif code == 20 and cur_x is not None:
                    verts.append((cur_x, float(val))); cur_x = None
                elif code == 70: closed = bool(int(val)&1)
                i += 2
            layer_seen[lyr] += 1
            if lyr in ignore_layers: continue
            if not verts or (closed and skip_closed): continue
            add_poly(verts, closed, nodes, pipes,
                     tol, orth_only, min_len)

        # --- POLYLINE / VERTEX / SEQEND ---
        elif tag == 'POLYLINE':
            hdr,i = collect_pairs(lines,i)
            lyr = hdr.get(8,'')
            closed = bool(int(hdr.get(70,'0')) & 1)
            verts: List[Tuple[float,float]] = []
            while i + 1 < n:
                if lines[i] != '0':
                    i += 1; continue
                sub = lines[i+1]; i += 2
                if sub == 'VERTEX':
                    vmap,i = collect_pairs(lines,i)
                    if 10 in vmap and 20 in vmap:
                        verts.append((float(vmap[10]), float(vmap[20])))
                elif sub == 'SEQEND':
                    break
            layer_seen[lyr] += 1
            if lyr in ignore_layers: continue
            if not verts or (closed and skip_closed): continue
            add_poly(verts, closed, nodes, pipes,
                     tol, orth_only, min_len)

        # any other entity: just skip
    return nodes.xy, pipes, layer_seen

def add_poly(verts, closed, nodes, pipes, tol, orth_only, min_len):
    for (x1,y1),(x2,y2) in zip(verts[:-1], verts[1:]):
        if keep_seg(x1,y1,x2,y2,tol=tol,orth_only=orth_only,min_len=min_len):
            pipes.append((nodes.index(x1,y1), nodes.index(x2,y2)))
    if closed and len(verts) > 2:
        x1,y1 = verts[-1]; x2,y2 = verts[0]
        if keep_seg(x1,y1,x2,y2,tol=tol,orth_only=orth_only,min_len=min_len):
            pipes.append((nodes.index(x1,y1), nodes.index(x2,y2)))

# ─────────────────────────────── plotting ────────────────────────────────
def show_plot(nodes, pipes, pressures=None):
    import matplotlib.pyplot as plt
    import numpy as np
    xy = np.asarray(nodes); seg = np.asarray(pipes)

    plt.figure(figsize=(6, 6))
    # pipes
    plt.plot(xy[seg.T, 0], xy[seg.T, 1], 'b-', lw=1.5, label='pipe')

    # nodes
    if pressures is None:
        plt.scatter(xy[:, 0], xy[:, 1], 30, c='red', label='node', zorder=5)
    else:
        p = np.asarray(pressures)
        sc = plt.scatter(xy[:, 0], xy[:, 1], 40, c=p,
                         cmap='viridis', edgecolor='k', zorder=5)
        plt.colorbar(sc, label='pressure [psi]')

    plt.axis('equal'); plt.grid(True)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.title('Irrigation DXF geometry'); plt.legend()
    plt.show()

# ───────────────────────────────── main ───────────────────────────────────
def main() -> None:
    opts = get_cli()
    lines = read_dxf_lines(Path(opts.dxf))

    if opts.list_layers:
        # quick layer dump & quit
        lyr = {lines[i+1] for i in range(0,len(lines)-1,2) if lines[i]=='8'}
        print('Layers found:', ', '.join(sorted(lyr) or ['(none)'])); return

    ignore = {s.strip() for s in opts.ignore_layers.split(',') if s.strip()}

    nodes,pipes,seen = parse_dxf(lines,
                                 ignore_layers=ignore,
                                 tol=opts.tol,
                                 min_len=opts.min_length,
                                 skip_closed=opts.skip_closed,
                                 orth_only=opts.orth_only)

    if opts.rebase and nodes:
        min_x = min(x for x,_ in nodes)
        min_y = min(y for _,y in nodes)
        nodes = [(x-min_x, y-min_y) for x,y in nodes]

    print(f'Imported {len(nodes)} nodes and {len(pipes)} pipes '
          f'from {opts.dxf}')
    if not pipes:
        print('\nNothing survived the filters. Entities seen per layer:')
        for lyr,cnt in seen.items():
            print(f'  {lyr or "(no layer)"} : {cnt}')
        sys.exit(0)

    pressures = None
    if opts.show_pressure:
        pressures = [None]*len(nodes)          # index by node id
        for i in range(0, len(lines)-1, 2):
            if lines[i] == '0' and lines[i+1] == 'TEXT':
                ent, j = collect_pairs(lines, i+2)
                lyr = ent.get(8, '')
                if lyr != 'PRESSURE': continue
                if not (10 in ent and 20 in ent and 1 in ent): continue
                x = float(ent[10]); y = float(ent[20])
                pval = float(ent[1])
                # match to nearest node
                best = min(range(len(nodes)),
                           key=lambda k: (nodes[k][0]-x)**2+(nodes[k][1]-y)**2)
                pressures[best] = pval
        # fallback: drop if none parsed
        if all(v is None for v in pressures):
            pressures = None

    if not opts.no_plot:
        show_plot(nodes, pipes, pressures if opts.show_pressure else None)

if __name__ == '__main__':
    main()
