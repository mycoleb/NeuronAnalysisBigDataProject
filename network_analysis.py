import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_network_metrics(matrix, regions):
    # Create network
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    
    # Calculate centrality measures
    centrality_metrics = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'eigenvector': nx.eigenvector_centrality_numpy(G),
        'pagerank': nx.pagerank(G)
    }
    
    # Calculate path lengths and efficiency
    avg_path_length = nx.average_shortest_path_length(G)
    global_efficiency = nx.global_efficiency(G)
    
    # Save results
    metrics_df = pd.DataFrame(centrality_metrics)
    metrics_df['region'] = regions
    metrics_df.to_csv('network_metrics.csv', index=False)
    
    print(f"Average Path Length: {avg_path_length:.3f}")
    print(f"Global Efficiency: {global_efficiency:.3f}")
    
    return centrality_metrics, avg_path_length, global_efficiency

def analyze_node_removal_impact(matrix, regions, top_n=10):
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    original_metrics = {
        'efficiency': nx.global_efficiency(G),
        'components': len(list(nx.strongly_connected_components(G))),
        'avg_path': nx.average_shortest_path_length(G)
    }
    
    impact_results = []
    for node in range(len(regions)):
        G_temp = G.copy()
        G_temp.remove_node(node)
        
        impact = {
            'node': regions[node],
            'efficiency_change': original_metrics['efficiency'] - nx.global_efficiency(G_temp),
            'component_change': len(list(nx.strongly_connected_components(G_temp))) - original_metrics['components']
        }
        impact_results.append(impact)
    
    # Convert to DataFrame and save
    impact_df = pd.DataFrame(impact_results)
    impact_df.to_csv('node_removal_impact.csv', index=False)
    
    # Plot top impactful nodes
    plt.figure(figsize=(12, 6))
    top_impacts = impact_df.nlargest(top_n, 'efficiency_change')
    sns.barplot(data=top_impacts, x='efficiency_change', y='node')
    plt.title(f'Top {top_n} Most Critical Nodes')
    plt.xlabel('Impact on Network Efficiency')
    plt.tight_layout()
    plt.savefig('critical_nodes.png')
    plt.close()
    
    return impact_df

def detect_communities(matrix, regions):
    G = nx.from_numpy_array(matrix, create_using=nx.Graph)
    communities = nx.community.greedy_modularity_communities(G)
    
    # Map communities to brain regions
    community_mapping = {}
    for i, community in enumerate(communities):
        for node in community:
            community_mapping[regions[node]] = i
    
    # Save community assignments
    community_df = pd.DataFrame({
        'region': regions,
        'community': [community_mapping[r] for r in regions]
    })
    community_df.to_csv('community_structure.csv', index=False)
    
    return communities, community_mapping

if __name__ == "__main__":
    # Load data
    matrix = np.load('connectivity_matrix.npy')
    with open('region_names.txt', 'r') as f:
        regions = [line.strip() for line in f]
    
    # Run analyses
    print("Calculating network metrics...")
    metrics, path_length, efficiency = analyze_network_metrics(matrix, regions)
    
    print("\nAnalyzing node removal impact...")
    impact_results = analyze_node_removal_impact(matrix, regions)
    
    print("\nDetecting communities...")
    communities, community_map = detect_communities(matrix, regions)
    
    print("\nAnalysis complete! Results saved to:")
    print("- network_metrics.csv")
    print("- node_removal_impact.csv")
    print("- community_structure.csv")
    print("- critical_nodes.png")