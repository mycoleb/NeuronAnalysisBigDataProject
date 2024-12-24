import numpy as np
from a import BrainConnectivityAnalyzer

# Load the saved data
connectivity_matrix = np.load('connectivity_matrix.npy')
with open('region_names.txt', 'r') as f:
    region_names = [line.strip() for line in f]

# Initialize the analyzer
analyzer = BrainConnectivityAnalyzer()
analyzer.load_connectivity_data(connectivity_matrix, region_names)

# Perform analyses
centrality_measures = analyzer.calculate_centrality_measures()
critical_nodes = analyzer.find_critical_nodes()
path_stats = analyzer.analyze_path_lengths()
network_metrics = analyzer.calculate_network_metrics()

# Print some key findings
print("\nNetwork Overview:")
print(f"Number of regions: {len(region_names)}")
print("\nNetwork Metrics:")
for metric, value in network_metrics.items():
    print(f"{metric}: {value:.3f}")

print("\nCritical Nodes:")
print(critical_nodes)

print("\nPath Length Statistics:")
for metric, value in path_stats.items():
    print(f"{metric}: {value:.3f}")

# Visualize the network
analyzer.visualize_network(
    highlight_nodes=critical_nodes,
    node_size_metric='betweenness',
    save_path='brain_network.png'
)