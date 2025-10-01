#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def read_data():
    """Read irrigation system data from stdin"""
    nodes = {}
    links = []
    current_section = None

    for line in sys.stdin:
        line = line.strip()
        if line == "NODES_START":
            current_section = "NODES"
            continue
        elif line == "NODES_END":
            current_section = None
            continue
        elif line == "LINKS_START":
            current_section = "LINKS"
            continue
        elif line == "LINKS_END":
            current_section = None
            continue

        if current_section == "NODES":
            parts = line.split()
            node_id = int(parts[0])
            nodes[node_id] = {
                'x': float(parts[1]),
                'y': float(parts[2]),
                'type': parts[3],
                'pressure': float(parts[4]),
                'is_fixed': bool(int(parts[5]))
            }
        elif current_section == "LINKS":
            parts = line.split()
            links.append({
                'from': int(parts[0]),
                'to': int(parts[1]),
                'diameter': float(parts[2]),
                'length': float(parts[3]),
                'type': parts[4]
            })

    return nodes, links


def visualize_complete_system(nodes, links, use_psi=True):
    """Visualize the irrigation system with pressure-based coloring and legend"""
    plt.figure(figsize=(18,12))
    ax = plt.gca()

    # --- Pressure color mapping ---
    pressures = [n['pressure'] for n in nodes.values() if n['pressure'] > 0]
    if pressures:
        min_p, max_p = min(pressures), max(pressures)
        norm = Normalize(min_p, max_p)
        cmap = plt.cm.plasma
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        units = "psi" if use_psi else "Pa"
        plt.colorbar(sm, ax=ax, label=f'Pressure ({units})', shrink=0.8, pad=0.02)
    else:
        cmap = plt.cm.plasma
        norm = None
        units = "psi" if use_psi else "Pa"

    # --- Draw pipes ---
    for link in links:
        if link['from'] in nodes and link['to'] in nodes:
            n1 = nodes[link['from']]
            n2 = nodes[link['to']]
            x = [n1['x'], n2['x']]
            y = [n1['y'], n2['y']]
            avg_p = 0.5 * (n1['pressure'] + n2['pressure'])
            color = cmap(norm(avg_p)) if norm else 'gray'
            ax.plot(x, y, color=color, linewidth=2, alpha=0.9)

    # --- Node markers ---
    type_markers = {
        "lateral_sprinkler_jn": ("o", 80),
        "lateral_sub_jn": ("o", 60),
        "barb": ("s", 50),
        "emitter": ("^", 50),
        "submain": ("D", 80),
        "waterSource": ("*", 200),
        "junction": ("o", 60),
    }

    for nid, n in nodes.items():
        marker, size = type_markers.get(n['type'], ("o", 50))
        node_color = cmap(norm(n['pressure'])) if norm else 'gray'
        ax.scatter(n['x'], n['y'], s=size, c=[node_color], marker=marker,
                   edgecolors='black', linewidths=0.8, zorder=3)

        ax.text(n['x'], n['y'] + 0.8, f"ID: {nid}",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))


        #  label water source and emitter nodes
        if n['type'] in ['waterSource', 'emitter']:
            ax.text(n['x'], n['y'] + 0.5, f"{n['pressure']:.1f} {units}",
                    fontsize=8, ha='center', va='bottom')



    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Sprinkler Junction',
               markerfacecolor='#2ecc71', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Submain Junction',
               markerfacecolor='#f39c12', markersize=12),
        Line2D([0], [0], marker='s', color='w', label='Barb',
               markerfacecolor='#9b59b6', markersize=12),
        Line2D([0], [0], marker='^', color='w', label='Emitter',
               markerfacecolor='#e74c3c', markersize=12),
        Line2D([0], [0], marker='*', color='w', label='Water Source',
               markerfacecolor='#3498db', markersize=18),
        Line2D([0], [0], color='#3498db', lw=2, label='Lateral Pipe'),
        Line2D([0], [0], color='#16a085', lw=2.5, label='Lateral to Barb'),
        Line2D([0], [0], color='#8e44ad', lw=2, ls='--', label='Barb to Emitter'),
        Line2D([0], [0], color='#f39c12', lw=3, label='Lateral to Submain'),
        Line2D([0], [0], color='#c0392b', lw=4, label='Submain Pipe'),
        Line2D([0], [0], color='#2c3e50', lw=4, label='Mainline')
    ]



    plt.title("Irrigation System Hydraulic Analysis Pressure Map", pad=20, fontsize=14, weight='bold')
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # --- Print only water source and emitter pressures ---
    print("\n--- Selected Node Pressures ---")
    for nid, n in nodes.items():
        if n['type'] in ['waterSource', 'emitter']:
            print(f"Node {nid} ({n['type']}): {n['pressure']:.2f} {units}")


if __name__ == "__main__":
    nodes, links = read_data()
    visualize_complete_system(nodes, links, use_psi=True)
