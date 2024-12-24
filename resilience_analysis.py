import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objects as go

class NetworkResilience:
    def __init__(self, matrix, regions):
        print("Initializing network analysis...")
        self.matrix = matrix
        self.regions = regions
        # Create both directed and undirected versions of the graph
        self.G_directed = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        self.G_undirected = nx.from_numpy_array(matrix, create_using=nx.Graph)
        self.baseline_metrics = self._calculate_baseline_metrics()
        
    def _calculate_baseline_metrics(self):
        """Calculate baseline network metrics before any perturbation."""
        print("Calculating baseline metrics...")
        metrics = {}
        
        # Metrics for undirected graph
        metrics['clustering_coefficient'] = nx.average_clustering(self.G_undirected)
        metrics['efficiency'] = nx.global_efficiency(self.G_undirected)
        
        # Metrics for directed graph
        metrics['strongly_connected_components'] = len(list(nx.strongly_connected_components(self.G_directed)))
        
        return metrics
    
    def analyze_node_importance(self):
        """Analyze importance of each node."""
        print("Analyzing node importance...")
        importance_metrics = {
            'degree': nx.degree_centrality(self.G_directed),
            'betweenness': nx.betweenness_centrality(self.G_directed),
            'pagerank': nx.pagerank(self.G_directed)
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(importance_metrics)
        df['region'] = self.regions
        return df
    
    def analyze_network_degradation(self, n_removals=50):
        """Analyze how network degrades with node removals."""
        print("Analyzing network degradation...")
        results = []
        
        # Get node importance for targeted removal
        importance = nx.betweenness_centrality(self.G_directed)
        nodes_by_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        G_temp = self.G_directed.copy()
        for i in tqdm(range(min(n_removals, len(nodes_by_importance)))):
            node_to_remove = nodes_by_importance[i][0]
            region_name = self.regions[node_to_remove]
            
            # Calculate metrics before removal
            metrics = {
                'step': i,
                'removed_region': region_name,
                'nodes_remaining': G_temp.number_of_nodes(),
                'edges_remaining': G_temp.number_of_edges(),
                'components': len(list(nx.strongly_connected_components(G_temp)))
            }
            
            # Remove node
            G_temp.remove_node(node_to_remove)
            results.append(metrics)
        
        return pd.DataFrame(results)

def visualize_degradation(df):
    """Create visualizations of network degradation."""
    print("Creating visualizations...")
    
    # Create static plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot nodes and edges remaining
    ax1 = axes[0]
    ax1.plot(df['step'], df['nodes_remaining'], 'b-', label='Nodes')
    ax1.plot(df['step'], df['edges_remaining'], 'r-', label='Edges')
    ax1.set_title('Network Size During Degradation')
    ax1.set_xlabel('Number of Nodes Removed')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True)
    
    # Plot number of components
    ax2 = axes[1]
    ax2.plot(df['step'], df['components'], 'g-')
    ax2.set_title('Network Fragmentation')
    ax2.set_xlabel('Number of Nodes Removed')
    ax2.set_ylabel('Number of Components')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('network_degradation.png')
    plt.close()
    
    # Create interactive plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['nodes_remaining'],
        name='Nodes Remaining',
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['edges_remaining'],
        name='Edges Remaining',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Interactive Network Degradation Analysis',
        xaxis_title='Nodes Removed',
        yaxis_title='Count',
        hovermode='x unified'
    )
    
    fig.write_html('degradation_analysis.html')

def main():
    print("Loading data...")
    matrix = np.load('connectivity_matrix.npy')
    with open('region_names.txt', 'r') as f:
        regions = [line.strip() for line in f]
    
    # Initialize analyzer
    analyzer = NetworkResilience(matrix, regions)
    
    # Analyze node importance
    importance_df = analyzer.analyze_node_importance()
    importance_df.to_csv('node_importance.csv', index=False)
    
    # Analyze network degradation
    degradation_df = analyzer.analyze_network_degradation()
    degradation_df.to_csv('network_degradation.csv', index=False)
    
    # Create visualizations
    visualize_degradation(degradation_df)
    
    print("\nAnalysis complete! Files created:")
    print("- node_importance.csv")
    print("- network_degradation.csv")
    print("- network_degradation.png")
    print("- degradation_analysis.html")
    
    # Print top 5 most important nodes
    print("\nTop 5 most important nodes (by betweenness centrality):")
    top_nodes = importance_df.nlargest(5, 'betweenness')
    print(top_nodes[['region', 'betweenness']].to_string())

if __name__ == "__main__":
    main()