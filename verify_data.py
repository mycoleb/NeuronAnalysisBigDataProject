import sqlite3
import numpy as np

def verify_database():
    # Connect to the database
    conn = sqlite3.connect('brain_connectivity.db')
    c = conn.cursor()
    
    # Check regions table
    c.execute('SELECT COUNT(*) FROM regions')
    region_count = c.fetchone()[0]
    print(f"\nNumber of regions in database: {region_count}")
    
    # Check connectivity table
    c.execute('SELECT COUNT(*) FROM connectivity')
    connection_count = c.fetchone()[0]
    print(f"Number of connections in database: {connection_count}")
    
    # Check some actual connectivity values
    c.execute('''SELECT r1.name as source, r2.name as target, c.projection_density 
                 FROM connectivity c 
                 JOIN regions r1 ON c.source_id = r1.id 
                 JOIN regions r2 ON c.target_id = r2.id 
                 LIMIT 5''')
    print("\nSample connections:")
    for row in c.fetchall():
        print(f"{row[0]} -> {row[1]}: {row[2]}")
    
    # Check the saved numpy matrix
    matrix = np.load('connectivity_matrix.npy')
    print(f"\nMatrix shape: {matrix.shape}")
    print(f"Non-zero connections: {np.count_nonzero(matrix)}")
    print(f"Average connection strength: {np.mean(matrix[matrix > 0]):.6f}")
    
    conn.close()

if __name__ == "__main__":
    verify_database()