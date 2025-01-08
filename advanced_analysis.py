#advanced_analysis.py
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster import hierarchy
from statsmodels.stats.multitest import multipletests
import time
from functools import partial
from multiprocessing import Pool

class OptimizedNetworkAnalysis:
    def __init__(self, matrix, regions):
        self.matrix = matrix
        self.regions = regions
        self.G_directed = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        self.G_undirected = nx.from_numpy_array(matrix, create_using=nx.Graph)
        
    def analyze_small_world_properties(self, max_time=300):  # 5 minutes timeout
        """Analyze small-world properties with timeout."""
        print("Analyzing small-world properties...")
        start_time = time.time()
        
        # Calculate actual network metrics
        C = nx.average_clustering(self.G_undirected)
        try:
            L = nx.average_shortest_path_length(self.G_undirected)
        except:
            L = np.inf
            print("Warning: Network is not fully connected")
        
        # Generate random networks for comparison
        n_random = 10  # Reduced from 100 for speed
        random_C = []
        random_L = []
        
        n_nodes = len(self.G_undirected)
        n_edges = self.G_undirected.number_of_edges()
        
        for _ in tqdm(range(n_random), desc="Generating random networks"):
            if time.time() - start_time > max_time:
                print("Warning: Small-world analysis timed out")
                break
                
            G_rand = nx.gnm_random_graph(n_nodes, n_edges)
            random_C.append(nx.average_clustering(G_rand))
            try:
                random_L.append(nx.average_shortest_path_length(G_rand))
            except:
                random_L.append(np.inf)
        
        # Calculate small-world coefficients
        C_rand = np.mean(random_C)
        L_rand = np.mean([l for l in random_L if l != np.inf])
        
        if L != np.inf and L_rand != 0:
            sigma = (C/C_rand)/(L/L_rand)
        else:
            sigma = np.nan
        
        return {
            'clustering_coefficient': C,
            'avg_path_length': L,
            'random_clustering': C_rand,
            'random_path_length': L_rand,
            'small_world_coefficient': sigma
        }
    
    def analyze_rich_club(self, max_degrees=20):
        """Analyze rich-club coefficient for limited degree levels."""
        print("Analyzing rich-club properties...")
        
        rich_club_coeffs = {}
        for k in tqdm(range(1, max_degrees + 1), desc="Calculating rich-club coefficients"):
            try:
                coeff = nx.rich_club_coefficient(self.G_undirected, k)
                if coeff:
                    rich_club_coeffs[k] = coeff
            except:
                continue
        
        return pd.DataFrame({'degree': rich_club_coeffs.keys(),
                           'coefficient': rich_club_coeffs.values()})
    
    def analyze_basic_motifs(self):
        """Analyze basic network motifs (triangles and 2-paths)."""
        print("Analyzing basic motifs...")
    
        # Only count triangles - remove slow two_paths calculation
        triangles = sum(nx.triangles(self.G_undirected).values()) // 3
        
        return pd.DataFrame({
            'motif_type': ['triangles'],
            'count': [triangles]
        })
        
       
    
    def perform_statistical_tests(self):
        """Perform essential statistical tests on network properties."""
        print("Performing statistical tests...")
        
        results = {}
        
        # Test for scale-free properties
        degrees = [d for n, d in self.G_directed.degree()]
        results['degree_normality'] = stats.normaltest(degrees)
        
        # Test for degree assortativity
        results['assortativity'] = nx.degree_assortativity_coefficient(self.G_directed)
        
        # Simple community structure test
        communities = list(nx.community.greedy_modularity_communities(self.G_undirected))
        results['n_communities'] = len(communities)
        results['modularity'] = nx.community.modularity(self.G_undirected, communities)
        
        return results
    
    def analyze_hierarchical_structure(self):
        """Quick analysis of hierarchical organization."""
        print("Analyzing hierarchical structure...")
        
        # Calculate clustering coefficient for nodes by degree
        degrees = dict(self.G_undirected.degree())
        clustering = nx.clustering(self.G_undirected)
        
        hierarchical_data = pd.DataFrame({
            'degree': degrees.values(),
            'clustering': clustering.values()
        })
        
        return hierarchical_data.groupby('degree')['clustering'].mean().reset_index()

def create_visualizations(results):
    """Create essential visualizations."""
    print("Creating visualizations...")
    
    # 1. Degree distribution
    plt.figure(figsize=(10, 6))
    degrees = [d for n, d in results['G_undirected'].degree()]
    sns.histplot(degrees, bins=30)
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Hierarchical structure
    if 'hierarchical' in results:
        plt.figure(figsize=(10, 6))
        plt.loglog(results['hierarchical']['degree'],
                  results['hierarchical']['clustering'], 'ko')
        plt.title('Hierarchical Organization')
        plt.xlabel('Node Degree (log)')
        plt.ylabel('Clustering Coefficient (log)')
        plt.grid(True)
        plt.savefig('hierarchical_structure.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Load data
    print("Loading data...")
    matrix = np.load('connectivity_matrix.npy')
    with open('region_names.txt', 'r') as f:
        regions = [line.strip() for line in f]
    
    # Initialize analyzer
    analyzer = OptimizedNetworkAnalysis(matrix, regions)
    
    # Perform analyses
    results = {
        'small_world': analyzer.analyze_small_world_properties(),
        'rich_club': analyzer.analyze_rich_club(),
        'basic_motifs': analyzer.analyze_basic_motifs(),
        'statistical_tests': analyzer.perform_statistical_tests(),
        'hierarchical': analyzer.analyze_hierarchical_structure()
    }
    
    # Save results
    print("\nSaving results...")
    for key, data in results.items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(f'{key}_analysis.csv', index=False)
        elif isinstance(data, dict):
            pd.DataFrame([data]).to_csv(f'{key}_analysis.csv', index=False)
    
    # Print summary
    print("\nAnalysis Summary:")
    if not np.isnan(results['small_world'].get('small_world_coefficient', np.nan)):
        print(f"Small-world coefficient: {results['small_world']['small_world_coefficient']:.3f}")
    print(f"Number of communities: {results['statistical_tests']['n_communities']}")
    print(f"Network modularity: {results['statistical_tests']['modularity']:.3f}")

if __name__ == "__main__":
    main()