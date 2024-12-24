import numpy as np
import sqlite3
import pandas as pd

def analyze_connectivity_patterns():
    # Load data
    conn = sqlite3.connect('brain_connectivity.db')
    matrix = np.load('connectivity_matrix.npy')
    with open('region_names.txt', 'r') as f:
        regions = [line.strip() for line in f]
    
    # Calculate key metrics
    total_connections = np.count_nonzero(matrix)
    avg_strength = np.mean(matrix[matrix > 0])
    
    # Find strongest connections
    n_top = 10
    flat_indices = np.argsort(matrix.flatten())[-n_top:]
    row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)
    
    print("Top Connections:")
    print("-" * 80)
    for i, (row, col) in enumerate(zip(row_indices, col_indices)):
        print(f"{i+1}. {regions[row]} -> {regions[col]}: {matrix[row, col]:.6f}")
    
    # Analyze region connectivity
    out_degree = np.sum(matrix > 0, axis=1)
    in_degree = np.sum(matrix > 0, axis=0)
    total_strength_out = np.sum(matrix, axis=1)
    total_strength_in = np.sum(matrix, axis=0)
    
    # Create region summary
    region_stats = pd.DataFrame({
        'Region': regions,
        'Outgoing_Connections': out_degree,
        'Incoming_Connections': in_degree,
        'Total_Outgoing_Strength': total_strength_out,
        'Total_Incoming_Strength': total_strength_in
    })
    
    print("\nMost Connected Regions (by number of connections):")
    print("-" * 80)
    top_connected = region_stats.nlargest(10, 'Outgoing_Connections')
    print(top_connected[['Region', 'Outgoing_Connections', 'Incoming_Connections']].to_string())
    
    print("\nStrongest Regions (by connection strength):")
    print("-" * 80)
    top_strength = region_stats.nlargest(10, 'Total_Outgoing_Strength')
    print(top_strength[['Region', 'Total_Outgoing_Strength', 'Total_Incoming_Strength']].to_string())
    
    # Save detailed statistics
    region_stats.to_csv('connectivity_statistics.csv', index=False)
    
    return region_stats

if __name__ == "__main__":
    stats = analyze_connectivity_patterns()