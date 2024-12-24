import numpy as np
import sqlite3
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor


def setup_database():
    """Create SQLite database and tables with indexes for faster queries."""
    conn = sqlite3.connect('brain_connectivity.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS regions
                 (id INTEGER PRIMARY KEY, 
                  name TEXT, 
                  atlas_id INTEGER)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_regions_atlas_id ON regions (atlas_id)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS connectivity
                 (source_id INTEGER,
                  target_id INTEGER,
                  projection_density REAL,
                  experiment_id INTEGER,
                  FOREIGN KEY (source_id) REFERENCES regions(id),
                  FOREIGN KEY (target_id) REFERENCES regions(id))''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_connectivity_source_target 
                 ON connectivity (source_id, target_id)''')
    
    conn.commit()
    return conn


def process_experiment(exp, mcc, atlas_to_db_id, conn):
    """Process a single experiment and save its data."""
    c = conn.cursor()
    try:
        exp_id = exp['id']
        source_id = exp['structure_id']
        unionizes = mcc.get_structure_unionizes([exp_id])
        if isinstance(unionizes, pd.DataFrame) and not unionizes.empty:
            rows_to_insert = []
            for _, union in unionizes.iterrows():
                db_source_id = atlas_to_db_id.get(source_id)
                db_target_id = atlas_to_db_id.get(union['structure_id'])
                density = union['projection_density']
                
                if all(v is not None for v in [db_source_id, db_target_id, density]):
                    rows_to_insert.append((db_source_id, db_target_id, float(density), exp_id))
            
            if rows_to_insert:
                c.executemany('''INSERT INTO connectivity VALUES (?, ?, ?, ?)''', rows_to_insert)
            conn.commit()
    except Exception as e:
        print(f"Error processing experiment {exp_id}: {e}")


def fetch_connectivity_data():
    """Fetch connectivity data and process it efficiently."""
    mcc = MouseConnectivityCache(manifest_file='mouse_connectivity_manifest.json')
    conn = setup_database()
    c = conn.cursor()
    
    # Get structure tree and isocortex structures
    structure_tree = mcc.get_structure_tree()
    structures = structure_tree.get_structures_by_set_id([167587189])
    print(f"Found {len(structures)} regions")
    
    # Store regions in the database
    atlas_to_db_id = {s['id']: i for i, s in enumerate(structures)}
    region_rows = [(i, s['name'], s['id']) for i, s in enumerate(structures)]
    c.executemany('INSERT OR REPLACE INTO regions VALUES (?, ?, ?)', region_rows)
    conn.commit()
    
    # Get experiments
    experiments = pd.DataFrame(mcc.get_experiments(dataframe=True))
    structure_ids = [s['id'] for s in structures]
    relevant_experiments = experiments[experiments['structure_id'].isin(structure_ids)]
    print(f"Processing {len(relevant_experiments)} experiments...")
    
    # Process experiments in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        for _, exp in tqdm(relevant_experiments.iterrows(), total=len(relevant_experiments)):
            executor.submit(process_experiment, exp, mcc, atlas_to_db_id, conn)
    
    # Create connectivity matrix
    c.execute('SELECT COUNT(*) FROM regions')
    n_regions = c.fetchone()[0]
    connectivity_matrix = np.zeros((n_regions, n_regions))
    c.execute('''SELECT source_id, target_id, AVG(projection_density) 
                 FROM connectivity 
                 GROUP BY source_id, target_id''')
    for source_id, target_id, density in c.fetchall():
        connectivity_matrix[source_id, target_id] = density
    
    # Save matrix and region names
    c.execute('SELECT name FROM regions ORDER BY id')
    region_names = [row[0] for row in c.fetchall()]
    np.save('connectivity_matrix.npy', connectivity_matrix)
    with open('region_names.txt', 'w') as f:
        f.writelines(f"{name}\n" for name in region_names)
    
    print("\nProcessing complete!")
    print(f"Data saved to brain_connectivity.db")
    print(f"Matrix shape: {connectivity_matrix.shape}")
    
    conn.close()
    return connectivity_matrix, region_names


if __name__ == "__main__":
    connectivity_matrix, region_names = fetch_connectivity_data()
