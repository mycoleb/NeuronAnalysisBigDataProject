import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objects as go
from scipy import stats

class ExpandedNodeAnalysis:
    def __init__(self, matrix, regions):
        self.matrix = matrix
        self.regions = regions
        # Create both directed and undirected versions
        self.G_directed = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        self.G_undirected = nx.from_numpy_array(matrix, create_using=nx.Graph)
        self.baseline_metrics = self._calculate_baseline()
        
    def _calculate_baseline(self):
        """Calculate baseline network metrics."""
        return {
            'efficiency': nx.global_efficiency(self.G_undirected),  # Use undirected for efficiency
            'components': len(list(nx.strongly_connected_components(self.G_directed))),
            'avg_clustering': nx.average_clustering(self.G_undirected),  # Use undirected for clustering
            'density': nx.density(self.G_directed),
            'pagerank': nx.pagerank(self.G_directed)
        }
    
    def analyze_regional_impact(self):
        """Analyze impact of removing nodes by brain region."""
        impacts = []
        
        print("Analyzing regional impact...")
        for idx, region in enumerate(tqdm(self.regions)):
            G_temp_dir = self.G_directed.copy()
            G_temp_undir = self.G_undirected.copy()
            
            G_temp_dir.remove_node(idx)
            G_temp_undir.remove_node(idx)
            
            # Calculate metrics using appropriate graph type
            impact = {
                'region': region,
                'efficiency_change': (nx.global_efficiency(G_temp_undir) - self.baseline_metrics['efficiency']) / self.baseline_metrics['efficiency'],
                'clustering_change': (nx.average_clustering(G_temp_undir) - self.baseline_metrics['avg_clustering']) / self.baseline_metrics['avg_clustering'],
                'density_change': (nx.density(G_temp_dir) - self.baseline_metrics['density']) / self.baseline_metrics['density'],
                'component_change': len(list(nx.strongly_connected_components(G_temp_dir))) - self.baseline_metrics['components']
            }
            impacts.append(impact)
        
        return pd.DataFrame(impacts)
    
    def analyze_cascading_failure(self, n_steps=10):
        """Analyze cascading failure starting from each node."""
        print("Analyzing cascading failures...")
        cascade_results = []
        
        for start_node in tqdm(range(len(self.regions))):
            G_temp_dir = self.G_directed.copy()
            G_temp_undir = self.G_undirected.copy()
            removed_nodes = [start_node]
            metrics_history = []
            
            for step in range(n_steps):
                if len(G_temp_dir) < 2:
                    break
                    
                # Remove most connected neighbor of previously removed nodes
                neighbors = set()
                for node in removed_nodes:
                    if node in G_temp_dir:
                        neighbors.update(G_temp_dir.neighbors(node))
                
                if not neighbors:
                    break
                
                # Find most connected neighbor
                neighbor_degrees = {n: G_temp_dir.degree(n) for n in neighbors}
                next_node = max(neighbor_degrees.items(), key=lambda x: x[1])[0]
                
                # Remove node from both graphs
                G_temp_dir.remove_node(next_node)
                G_temp_undir.remove_node(next_node)
                removed_nodes.append(next_node)
                
                metrics = {
                    'start_region': self.regions[start_node],
                    'step': step + 1,
                    'nodes_removed': len(removed_nodes),
                    'efficiency': nx.global_efficiency(G_temp_undir),
                    'components': len(list(nx.strongly_connected_components(G_temp_dir)))
                }
                metrics_history.append(metrics)
            
            cascade_results.extend(metrics_history)
        
        return pd.DataFrame(cascade_results)
    
    def analyze_targeted_vs_random(self, n_removals=50, n_random_trials=10):
        """Compare targeted vs random node removal."""
        print("Comparing targeted vs random removal...")
        results = []
        
        # Targeted removal (by degree)
        G_temp_dir = self.G_directed.copy()
        G_temp_undir = self.G_undirected.copy()
        
        for i in range(n_removals):
            if len(G_temp_dir) < 2:
                break
            
            # Remove highest degree node
            node_degrees = dict(G_temp_dir.degree())
            node_to_remove = max(node_degrees.items(), key=lambda x: x[1])[0]
            
            metrics = {
                'strategy': 'targeted',
                'step': i,
                'efficiency': nx.global_efficiency(G_temp_undir),
                'components': len(list(nx.strongly_connected_components(G_temp_dir)))
            }
            results.append(metrics)
            
            G_temp_dir.remove_node(node_to_remove)
            G_temp_undir.remove_node(node_to_remove)
        
        # Random removal (multiple trials)
        for trial in range(n_random_trials):
            G_temp_dir = self.G_directed.copy()
            G_temp_undir = self.G_undirected.copy()
            nodes = list(G_temp_dir.nodes())
            np.random.shuffle(nodes)
            
            for i in range(n_removals):
                if len(G_temp_dir) < 2:
                    break
                
                metrics = {
                    'strategy': f'random_trial_{trial}',
                    'step': i,
                    'efficiency': nx.global_efficiency(G_temp_undir),
                    'components': len(list(nx.strongly_connected_components(G_temp_dir)))
                }
                results.append(metrics)
                
                G_temp_dir.remove_node(nodes[i])
                G_temp_undir.remove_node(nodes[i])
        
        return pd.DataFrame(results)

def create_visualizations(regional_impact_df, cascade_df, comparison_df):
    """Create visualizations for all analyses."""
    print("Creating visualizations...")
    
    # 1. Regional Impact Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(regional_impact_df['efficiency_change'], bins=30)
    plt.title('Distribution of Regional Impact on Network Efficiency')
    plt.xlabel('Efficiency Change')
    plt.ylabel('Count')
    plt.savefig('regional_impact_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cascading Failure Progress
    fig = go.Figure()
    
    # Get top 5 most disruptive regions
    top_regions = cascade_df.groupby('start_region')['efficiency'].mean().nlargest(5).index
    
    for region in top_regions:
        region_data = cascade_df[cascade_df['start_region'] == region]
        fig.add_trace(go.Scatter(
            x=region_data['step'],
            y=region_data['efficiency'],
            name=region,
            mode='lines',
        ))
    
    fig.update_layout(
        title='Cascading Failure Progression (Top 5 Most Disruptive Regions)',
        xaxis_title='Steps',
        yaxis_title='Network Efficiency',
        showlegend=True
    )
    fig.write_html('cascade_progression.html')
    
    # 3. Targeted vs Random Comparison
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=comparison_df[comparison_df['strategy'] == 'targeted'], 
                x='step', y='efficiency', label='Targeted', color='red')
    sns.lineplot(data=comparison_df[comparison_df['strategy'].str.startswith('random')],
                x='step', y='efficiency', label='Random (avg)', color='blue')
    plt.title('Targeted vs Random Node Removal')
    plt.xlabel('Nodes Removed')
    plt.ylabel('Network Efficiency')
    plt.savefig('targeted_vs_random.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    matrix = np.load('connectivity_matrix.npy')
    with open('region_names.txt', 'r') as f:
        regions = [line.strip() for line in f]
    
    # Initialize analyzer
    analyzer = ExpandedNodeAnalysis(matrix, regions)
    
    # Run analyses
    print("\nRunning analyses...")
    regional_impact = analyzer.analyze_regional_impact()
    cascade_results = analyzer.analyze_cascading_failure()
    removal_comparison = analyzer.analyze_targeted_vs_random()
    
    # Save results
    print("\nSaving results...")
    regional_impact.to_csv('expanded_regional_impact.csv', index=False)
    cascade_results.to_csv('cascade_analysis.csv', index=False)
    removal_comparison.to_csv('removal_comparison.csv', index=False)
    
    # Create visualizations
    create_visualizations(regional_impact, cascade_results, removal_comparison)
    
    print("\nAnalysis complete! Files created:")
    print("- expanded_regional_impact.csv")
    print("- cascade_analysis.csv")
    print("- removal_comparison.csv")
    print("- regional_impact_dist.png")
    print("- cascade_progression.html")
    print("- targeted_vs_random.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nTop 5 Most Critical Regions (by efficiency impact):")
    print(regional_impact.nlargest(5, 'efficiency_change')[['region', 'efficiency_change']])

if __name__ == "__main__":
    main()