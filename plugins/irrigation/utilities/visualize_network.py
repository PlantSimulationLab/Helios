#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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

def visualize_complete_system(nodes, links):
    """Visualize the irrigation system with pressures and optional color-coding"""
    plt.figure(figsize=(18, 14))
    ax = plt.gca()

    # Visual styles for all components
    styles = {
        'nodes': {
            'lateral_sprinkler_jn': {'color': '#2ecc71', 'marker': 'o', 'size': 100, 'zorder': 5},
            'lateral_sub_jn': {'color': '#f39c12', 'marker': 'o', 'size': 120, 'zorder': 5},
            'barb': {'color': '#9b59b6', 'marker': 's', 'size': 80, 'zorder': 5},
            'emitter': {'color': '#e74c3c', 'marker': '^', 'size': 80, 'zorder': 5},
            'submain': {'color': '#c0392b', 'marker': 'D', 'size': 100, 'zorder': 4},
            'waterSource': {'color': '#3498db', 'marker': '*', 'size': 150, 'zorder': 6}
        },
        'links': {
            'lateral': {'color': '#3498db', 'lw': 2, 'zorder': 1},
            'lateralTobarb': {'color': '#16a085', 'lw': 2.5, 'zorder': 2},
            'barbToemitter': {'color': '#8e44ad', 'lw': 2, 'ls': '--', 'zorder': 3},
            'lateralToSubmain': {'color': '#f39c12', 'lw': 3, 'zorder': 2},
            'submainConnection': {'color': '#d35400', 'lw': 3, 'zorder': 2},
            'submain': {'color': '#c0392b', 'lw': 4, 'zorder': 1},
            'mainline': {'color': '#2c3e50', 'lw': 4, 'zorder': 1}
        }
    }

    # --- Pressure Color Mapping (Optional) ---
    pressures = [node['pressure'] for node in nodes.values() if node['pressure'] > 0]
    if pressures:
        min_p, max_p = min(pressures), max(pressures)
        norm = Normalize(min_p, max_p)
        cmap = plt.cm.plasma  # Alternative: plt.cm.viridis
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Pressure (Pa)', shrink=0.8)

    # Draw all links as straight lines
    for link in links:
        if link['from'] in nodes and link['to'] in nodes:
            from_node = nodes[link['from']]
            to_node = nodes[link['to']]
            style = styles['links'].get(link['type'], {})

            # Draw straight line
            ax.plot([from_node['x'], to_node['x']],
                    [from_node['y'], to_node['y']],
                    **style)

            # Label with length
            mid_x = (from_node['x'] + to_node['x']) / 2
            mid_y = (from_node['y'] + to_node['y']) / 2
            ax.text(mid_x, mid_y, f"{link['length']:.2f}m",
                    ha='center', va='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw nodes with pressure labels
    for node_id, node in nodes.items():
        style = styles['nodes'].get(node['type'], {'color': 'gray', 'marker': 'o', 'size': 80})

        # Dynamic color if pressure > 0 (optional)
        node_color = cmap(norm(node['pressure'])) if ('cmap' in locals() and node['pressure'] > 0) else style['color']

        ax.scatter(node['x'], node['y'],
                   marker=style['marker'],
                   c=node_color,
                   s=style['size'],
                   zorder=style['zorder'],
                   edgecolors='k',
                   linewidths=0.8)

        # Add pressure label (convert Pa to psi)
        pressure_psi = node['pressure'] / 6894.76 if node['pressure'] != 0 else 0
        label = f"{pressure_psi:.1f} psi"

        # Smart label positioning
        offset_x, offset_y = 0, 0
        if node['type'] == 'emitter':
            offset_y = -0.4
        elif node['type'] == 'barb':
            offset_y = 0.4
        elif node['type'] == 'lateral_sprinkler_jn':
            offset_x = 0.4

        ax.text(node['x'] + offset_x,
                node['y'] + offset_y,
                label,
                ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, pad=2, edgecolor='none'),
                zorder=10)

        # Label key nodes
        if node['type'] in ['waterSource', 'lateral_sub_jn']:
            offset = 0.4 if node['type'] == 'waterSource' else 0.3
            ax.text(node['x'], node['y'] + offset, node['type'],
                    ha='center', va='bottom', fontsize=10,
                    weight='bold', zorder=10,
                    bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Sprinkler Junction',
               markerfacecolor='#2ecc71', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Submain Junction',
               markerfacecolor='#f39c12', markersize=12),
        Line2D([0], [0], marker='s', color='w', label='Barb',
               markerfacecolor='#9b59b6', markersize=12),
        Line2D([0], [0], marker='^', color='w', label='Emitter',
               markerfacecolor='#e74c3c', markersize=12),
        Line2D([0], [0], marker='D', color='w', label='Submain',
               markerfacecolor='#c0392b', markersize=12),
        Line2D([0], [0], marker='*', color='w', label='Water Source',
               markerfacecolor='#3498db', markersize=18),
        Line2D([0], [0], color='#3498db', lw=2, label='Lateral Pipe'),
        Line2D([0], [0], color='#16a085', lw=2.5, label='Lateral to Barb'),
        Line2D([0], [0], color='#8e44ad', lw=2, ls='--', label='Barb to Emitter'),
        Line2D([0], [0], color='#f39c12', lw=3, label='Lateral to Submain'),
        Line2D([0], [0], color='#c0392b', lw=4, label='Submain Pipe'),
        Line2D([0], [0], color='#2c3e50', lw=4, label='Mainline')
    ]

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.35, 1), fontsize=10, title="System Components",
              title_fontsize=11, framealpha=0.9)

    plt.title("Irrigation System Hydraulic Analysis\n(Pressures in psi, Pipe Lengths in meters)",
              pad=20, fontsize=14, weight='bold')
    plt.xlabel("X Position (meters)", fontsize=12)
    plt.ylabel("Y Position (meters)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nodes, links = read_data()
    visualize_complete_system(nodes, links)