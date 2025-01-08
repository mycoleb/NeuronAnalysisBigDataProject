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

def analyze_resilience(self):
    """
    Analyze network resilience through multiple metrics:
    - Attack tolerance (targeted vs random removal)
    - Topological resilience
    - Path redundancy
    - Critical node identification
    """
    print("Analyzing network resilience...")
    results = []
    
    # Analyze different attack strategies
    n_removals = min(50, len(self.regions))  # Don't remove more nodes than we have
    
    # 1. Random failure analysis
    for trial in range(5):  # Multiple trials for random removal
        G_temp = self.G_directed.copy()
        nodes = list(G_temp.nodes())
        np.random.shuffle(nodes)
        
        for i in range(n_removals):
            if len(G_temp) < 2:
                break
                
            metrics = self._calculate_resilience_metrics(G_temp, f'random_trial_{trial}', i)
            results.append(metrics)
            G_temp.remove_node(nodes[i])
    
    # 2. Targeted attack analysis (highest degree first)
    G_temp = self.G_directed.copy()
    nodes_by_degree = sorted(dict(G_temp.degree()).items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    
    for i in range(n_removals):
        if len(G_temp) < 2:
            break
            
        metrics = self._calculate_resilience_metrics(G_temp, 'targeted_attack', i)
        results.append(metrics)
        
        if nodes_by_degree:
            node_to_remove = nodes_by_degree.pop(0)[0]
            G_temp.remove_node(node_to_remove)
            
    return pd.DataFrame(results)

def _calculate_resilience_metrics(self, G, strategy, step):
    """Calculate comprehensive resilience metrics for the current network state."""
    metrics = {
        'strategy': strategy,
        'step': step,
        'nodes_remaining': G.number_of_nodes(),
        'edges_remaining': G.number_of_edges(),
        'components': len(list(nx.strongly_connected_components(G))),
        'avg_path_length': -1,  # Default value if calculation fails
        'efficiency': -1,
        'clustering': -1
    }
    
    # Calculate additional metrics safely
    try:
        # Convert to undirected for some metrics
        G_undir = G.to_undirected()
        
        # Network efficiency
        metrics['efficiency'] = nx.global_efficiency(G_undir)
        
        # Average clustering
        metrics['clustering'] = nx.average_clustering(G_undir)
        
        # Average path length (if network is still connected)
        if nx.is_strongly_connected(G):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
            
    except:
        pass  # Keep default values if calculations fail
        
    return metrics

def visualize_resilience(df):
    """Create interactive visualization of resilience analysis."""
    print("Creating resilience visualizations...")
    
    # Create interactive plot with multiple metrics
    fig = go.Figure()
    
    # Plot efficiency for different strategies
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        
        # Efficiency plot
        fig.add_trace(go.Scatter(
            x=strategy_data['step'],
            y=strategy_data['efficiency'],
            name=f'{strategy} - Efficiency',
            mode='lines',
            line=dict(dash='solid' if 'targeted' in strategy else 'dot')
        ))
    
    fig.update_layout(
        title='Network Resilience Analysis',
        xaxis_title='Number of Nodes Removed',
        yaxis_title='Global Efficiency',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save interactive visualization
    fig.write_html('resilience_analysis.html')
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
    
    # Analyze network resilience (new!)
    resilience_df = analyzer.analyze_resilience()
    resilience_df.to_csv('resilience_metrics.csv', index=False)
    
    # Create visualizations
    visualize_degradation(degradation_df)
    visualize_resilience(resilience_df)
    
    print("\nAnalysis complete! Files created:")
    print("- node_importance.csv")
    print("- network_degradation.csv")
    print("- network_degradation.png")
    print("- degradation_analysis.html")
    print("- resilience_analysis.html")
    print("- resilience_metrics.csv")
    
    # Print summary statistics
    print("\nResilience Summary:")
    baseline_efficiency = resilience_df[resilience_df['step'] == 0]['efficiency'].mean()
    final_efficiency_targeted = resilience_df[
        (resilience_df['strategy'] == 'targeted_attack') & 
        (resilience_df['step'] == resilience_df['step'].max())
    ]['efficiency'].iloc[0]
    
    print(f"Baseline network efficiency: {baseline_efficiency:.4f}")
    print(f"Efficiency after targeted attack: {final_efficiency_targeted:.4f}")
    print(f"Efficiency decrease: {((baseline_efficiency - final_efficiency_targeted)/baseline_efficiency)*100:.1f}%")

if __name__ == "__main__":
    main()