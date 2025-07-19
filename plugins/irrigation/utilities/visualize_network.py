#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def read_data():
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

def visualize_network(nodes, links):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Custom styling
    node_styles = {
        'lateral_sprinkler_jn': {'color': 'green', 'marker': 'o', 'size': 8},
        'barb': {'color': 'magenta', 'marker': 's', 'size': 7},
        'emitter': {'color': 'red', 'marker': '^', 'size': 8}
    }

    link_styles = {
        'lateral': {'color': 'blue', 'linewidth': 1.5, 'linestyle': '-'},
        'barbToemitter': {'color': 'purple', 'linewidth': 1, 'linestyle': '--'}
    }

    # Draw links first (background)
    for link in links:
        from_node = nodes[link['from']]
        to_node = nodes[link['to']]
        style = link_styles.get(link['type'], {'color': 'gray'})
        ax.plot([from_node['x'], to_node['x']],
                [from_node['y'], to_node['y']],
                **style)

    # Draw nodes on top
    for node_id, node in nodes.items():
        style = node_styles.get(node['type'], {'color': 'gray', 'marker': 'o', 'size': 6})
        ax.plot(node['x'], node['y'],
                marker=style['marker'],
                color=style['color'],
                markersize=style['size'])

        # Label nodes with their IDs
        ax.text(node['x'], node['y'], str(node_id),
                ha='center', va='center',
                fontsize=8, color='white')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Junction',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Barb',
               markerfacecolor='magenta', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Emitter',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='blue', lw=2, label='Lateral Pipe'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Barb to Emitter')
    ]

    ax.legend(handles=legend_elements, loc='upper right')

    plt.title("Irrigation System Network Visualization")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nodes, links = read_data()
    visualize_network(nodes, links)