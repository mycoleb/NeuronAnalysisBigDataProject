import numpy as np
from a import BrainConnectivityAnalyzer
import pandas as pd

def analyze_brain_connectivity():
    # Load the saved data
    connectivity_matrix = np.load('connectivity_matrix.npy')
    with open('region_names.txt', 'r') as f:
        region_names = [line.strip() for line in f]
        
    # Create a dictionary mapping region names to their indices
    region_indices = {name: idx for idx, name in enumerate(region_names)}
    
    # Initialize analyzer
    analyzer = BrainConnectivityAnalyzer()
    analyzer.load_connectivity_data(connectivity_matrix, region_names)
    
    # Calculate key metrics
    centrality_measures = analyzer.calculate_centrality_measures()
    critical_nodes = analyzer.find_critical_nodes()
    path_stats = analyzer.analyze_path_lengths()
    network_metrics = analyzer.calculate_network_metrics()
    
    # Create detailed analysis results
    results = {
        'network_overview': {
            'total_regions': len(region_names),
            'total_connections': np.count_nonzero(connectivity_matrix),
            'average_connectivity': np.mean(connectivity_matrix),
            'max_connectivity': np.max(connectivity_matrix)
        },
        'region_metrics': {},
        'critical_regions': critical_nodes,
        'network_metrics': network_metrics,
        'path_statistics': path_stats
    }
    
    # Calculate per-region statistics
    for region in region_names:
        idx = region_indices[region]
        results['region_metrics'][region] = {
            'outgoing_connections': np.count_nonzero(connectivity_matrix[idx, :]),
            'incoming_connections': np.count_nonzero(connectivity_matrix[:, idx]),
            'strongest_output': np.max(connectivity_matrix[idx, :]),
            'strongest_input': np.max(connectivity_matrix[:, idx]),
            'degree_centrality': centrality_measures['degree'][region],
            'betweenness_centrality': centrality_measures['betweenness'][region],
            'is_critical_node': region in critical_nodes
        }
    
    # Save detailed results
    save_analysis_results(results)
    
    # Visualize with region grouping
    analyzer.visualize_network(
        highlight_nodes=critical_nodes,
        node_size_metric='betweenness',
        save_path='brain_network_detailed.png'
    )
    
    return results

def save_analysis_results(results):
    # Save network overview
    with open('analysis_results.txt', 'w') as f:
        f.write("Brain Connectivity Analysis Results\n")
        f.write("==================================\n\n")
        
        # Network overview
        f.write("Network Overview:\n")
        for metric, value in results['network_overview'].items():
            f.write(f"{metric}: {value}\n")
        
        # Critical regions
        f.write("\nCritical Regions:\n")
        for region in results['critical_regions']:
            f.write(f"- {region}\n")
            
        # Detailed region metrics
        f.write("\nDetailed Region Metrics:\n")
        for region, metrics in results['region_metrics'].items():
            f.write(f"\n{region}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
    
    # Save metrics as CSV for further analysis
    metrics_df = pd.DataFrame.from_dict(results['region_metrics'], orient='index')
    metrics_df.to_csv('region_metrics.csv')

if __name__ == "__main__":
    results = analyze_brain_connectivity()
    print("Analysis complete! Check analysis_results.txt and region_metrics.csv for detailed results.")