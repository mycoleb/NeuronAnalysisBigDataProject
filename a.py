import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

class BrainConnectivityAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.regions = {}
        
    def load_connectivity_data(self, connectivity_matrix: np.ndarray, region_names: List[str]) -> None:
        """
        Load connectivity data from a matrix and region names.
        
        Args:
            connectivity_matrix: NxN numpy array where N is the number of regions
            region_names: List of region names corresponding to matrix indices
        """
        self.regions = {i: name for i, name in enumerate(region_names)}
        
        # Create weighted edges between regions
        for i in range(len(region_names)):
            for j in range(len(region_names)):
                if connectivity_matrix[i][j] > 0:
                    self.graph.add_edge(
                        region_names[i],
                        region_names[j],
                        weight=connectivity_matrix[i][j]
                    )
    
    def calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate various centrality measures for each node.
        
        Returns:
            Dictionary containing different centrality measures for each node
        """
        centrality_measures = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality_numpy(self.graph),
            'pagerank': nx.pagerank(self.graph)
        }
        return centrality_measures
    
    def find_critical_nodes(self, metric: str = 'betweenness', threshold: float = 0.9) -> List[str]:
        """
        Identify critical nodes based on centrality measures.
        
        Args:
            metric: Centrality measure to use ('degree', 'betweenness', 'eigenvector', 'pagerank')
            threshold: Percentile threshold for considering nodes as critical
            
        Returns:
            List of critical node names
        """
        centrality = self.calculate_centrality_measures()[metric]
        threshold_value = np.percentile(list(centrality.values()), threshold * 100)
        return [node for node, value in centrality.items() if value >= threshold_value]
    
    def analyze_path_lengths(self) -> Dict[str, float]:
        """
        Analyze shortest path lengths between all pairs of nodes.
        
        Returns:
            Dictionary containing path length statistics
        """
        path_lengths = []
        for source in self.graph.nodes():
            for target in self.graph.nodes():
                if source != target:
                    try:
                        length = nx.shortest_path_length(
                            self.graph, source, target, weight='weight'
                        )
                        path_lengths.append(length)
                    except nx.NetworkXNoPath:
                        continue
        
        return {
            'average_path_length': np.mean(path_lengths),
            'max_path_length': np.max(path_lengths),
            'min_path_length': np.min(path_lengths)
        }
    
    def simulate_node_removal(self, node: str) -> Dict[str, float]:
        """
        Simulate the impact of removing a specific node on network metrics.
        
        Args:
            node: Name of the node to remove
            
        Returns:
            Dictionary containing network metrics after node removal
        """
        temp_graph = self.graph.copy()
        temp_graph.remove_node(node)
        
        # Calculate network metrics before and after removal
        original_metrics = self.calculate_network_metrics()
        
        # Temporarily set graph to modified version
        original_graph = self.graph
        self.graph = temp_graph
        new_metrics = self.calculate_network_metrics()
        self.graph = original_graph
        
        return {
            'metric': {
                'before': original_metrics[k],
                'after': new_metrics[k]
            } for k in original_metrics.keys()
        }
    
    def calculate_network_metrics(self) -> Dict[str, float]:
        """
        Calculate various network-wide metrics.
        
        Returns:
            Dictionary containing network metrics
        """
        return {
            'density': nx.density(self.graph),
            'transitivity': nx.transitivity(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'number_strongly_connected_components': nx.number_strongly_connected_components(self.graph)
        }
    
    def visualize_network(self, 
                         highlight_nodes: List[str] = None,
                         node_size_metric: str = 'degree',
                         save_path: str = None) -> None:
        """
        Visualize the brain region network.
        
        Args:
            highlight_nodes: List of nodes to highlight
            node_size_metric: Metric to determine node sizes
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Calculate node sizes based on centrality
        centrality = self.calculate_centrality_measures()[node_size_metric]
        node_sizes = [centrality[node] * 3000 for node in self.graph.nodes()]
        
        # Calculate edge weights for width
        edge_weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges()]
        
        # Create layout
        pos = nx.spring_layout(self.graph)
        
        # Draw network
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, 
                             width=[w/max(edge_weights) * 2 for w in edge_weights])
        
        # Draw nodes
        node_colors = ['red' if node in (highlight_nodes or []) else 'lightblue' 
                      for node in self.graph.nodes()]
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                             node_size=node_sizes)
        
        # Add labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        plt.title("Brain Region Connectivity Network")
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Create sample data
    n_regions = 5
    sample_regions = [f"Region_{i}" for i in range(n_regions)]
    sample_connectivity = np.random.random((n_regions, n_regions))
    np.fill_diagonal(sample_connectivity, 0)  # No self-connections
    
    # Initialize analyzer
    analyzer = BrainConnectivityAnalyzer()
    
    # Load data
    analyzer.load_connectivity_data(sample_connectivity, sample_regions)
    
    # Calculate centrality measures
    centrality = analyzer.calculate_centrality_measures()
    print("Centrality measures:", centrality)
    
    # Find critical nodes
    critical_nodes = analyzer.find_critical_nodes()
    print("Critical nodes:", critical_nodes)
    
    # Visualize network
    analyzer.visualize_network(highlight_nodes=critical_nodes)