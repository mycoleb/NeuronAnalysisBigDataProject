import numpy as np
import sqlite3
import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path

class ConnectivityAnalyzer:
    def __init__(self, matrix_path: str = 'connectivity_matrix.npy',
                 regions_path: str = 'region_names.txt',
                 db_path: str = 'brain_connectivity.db'):
        """
        Initialize the ConnectivityAnalyzer with paths to required data files.
        
        Args:
            matrix_path: Path to the connectivity matrix .npy file
            regions_path: Path to the region names text file
            db_path: Path to the SQLite database
        """
        self.matrix_path = Path(matrix_path)
        self.regions_path = Path(regions_path)
        self.db_path = Path(db_path)
        
        # Load data
        self.matrix = self._load_matrix()
        self.regions = self._load_regions()
        self.conn = self._connect_db()
        
    def _load_matrix(self) -> np.ndarray:
        """Load and verify the connectivity matrix."""
        try:
            matrix = np.load(self.matrix_path)
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Connectivity matrix must be square")
            return matrix
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find connectivity matrix at {self.matrix_path}")
            
    def _load_regions(self) -> List[str]:
        """Load and verify the region names."""
        try:
            with open(self.regions_path, 'r') as f:
                regions = [line.strip() for line in f]
            if len(regions) != self.matrix.shape[0]:
                raise ValueError("Number of regions doesn't match matrix dimensions")
            return regions
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find region names at {self.regions_path}")
            
    def _connect_db(self) -> sqlite3.Connection:
        """Establish database connection."""
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Database connection failed: {e}")

    def find_strongest_connections(self, n_top: int = 10) -> pd.DataFrame:
        """
        Find the strongest connections in the network.
        
        Args:
            n_top: Number of top connections to return
            
        Returns:
            DataFrame containing source, target, and strength of top connections
        """
        # Flatten and find top indices - FIXED to include the strongest connection
        flat_indices = np.argsort(self.matrix.flatten())[-n_top:][::-1]  # Removed the -1 slice
        row_indices, col_indices = np.unravel_index(flat_indices, self.matrix.shape)
        
        # Create DataFrame with results
        connections = pd.DataFrame({
            'Source': [self.regions[i] for i in row_indices],
            'Target': [self.regions[i] for i in col_indices],
            'Strength': self.matrix[row_indices, col_indices]
        })
        
        return connections

    def analyze_region_connectivity(self) -> pd.DataFrame:
        """
        Analyze connectivity patterns for each region.
        
        Returns:
            DataFrame with connectivity metrics for each region
        """
        # Calculate basic metrics
        out_degree = np.sum(self.matrix > 0, axis=1)
        in_degree = np.sum(self.matrix > 0, axis=0)
        total_strength_out = np.sum(self.matrix, axis=1)
        total_strength_in = np.sum(self.matrix, axis=0)
        
        # Calculate additional metrics
        avg_out_strength = np.mean(self.matrix, axis=1)
        avg_in_strength = np.mean(self.matrix, axis=0)
        max_out = np.max(self.matrix, axis=1)
        max_in = np.max(self.matrix, axis=0)
        
        return pd.DataFrame({
            'Region': self.regions,
            'Outgoing_Connections': out_degree,
            'Incoming_Connections': in_degree,
            'Total_Outgoing_Strength': total_strength_out,
            'Total_Incoming_Strength': total_strength_in,
            'Avg_Outgoing_Strength': avg_out_strength,
            'Avg_Incoming_Strength': avg_in_strength,
            'Max_Outgoing': max_out,
            'Max_Incoming': max_in
        })

    def get_network_statistics(self) -> Dict:
        """Calculate overall network statistics."""
        return {
            'total_connections': np.count_nonzero(self.matrix),
            'avg_strength': np.mean(self.matrix[self.matrix > 0]),
            'max_strength': np.max(self.matrix),
            'sparsity': np.count_nonzero(self.matrix) / self.matrix.size,
            'reciprocity': np.sum((self.matrix > 0) & (self.matrix.T > 0)) / np.sum(self.matrix > 0)
        }

def main():
    """Main execution function."""
    try:
        # Initialize analyzer
        analyzer = ConnectivityAnalyzer()
        
        # Get overall statistics
        stats = analyzer.get_network_statistics()
        print("\nNetwork Statistics:")
        print("-" * 80)
        for metric, value in stats.items():
            print(f"{metric}: {value:.6f}")
        
        # Get strongest connections
        print("\nTop Connections:")
        print("-" * 80)
        connections = analyzer.find_strongest_connections()
        for i, row in connections.iterrows():
            print(f"{i+1}. {row['Source']} -> {row['Target']}: {row['Strength']:.6f}")
        
        # Analyze region connectivity
        region_stats = analyzer.analyze_region_connectivity()
        
        # Show most connected regions
        print("\nMost Connected Regions (by number of connections):")
        print("-" * 80)
        print(region_stats.nlargest(10, 'Outgoing_Connections')[
            ['Region', 'Outgoing_Connections', 'Incoming_Connections']
        ].to_string())
        
        # Show strongest regions
        print("\nStrongest Regions (by connection strength):")
        print("-" * 80)
        print(region_stats.nlargest(10, 'Total_Outgoing_Strength')[
            ['Region', 'Total_Outgoing_Strength', 'Total_Incoming_Strength']
        ].to_string())
        
        # Save results
        connections.to_csv('top_connections.csv', index=False)
        region_stats.to_csv('connectivity_statistics.csv', index=False)
        pd.DataFrame([stats]).to_csv('network_statistics.csv', index=False)
        
        print("\nResults saved to CSV files.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()