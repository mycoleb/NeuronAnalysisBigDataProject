from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import pandas as pd

# Initialize cache
mcc = MouseConnectivityCache(manifest_file='mouse_connectivity_manifest.json')

# Get structure info for Frontal pole
structure_tree = mcc.get_structure_tree()
frontal_pole = structure_tree.get_structures_by_name(['Frontal pole, cerebral cortex'])[0]
frontal_pole_id = frontal_pole['id']

# Get experiments for Frontal pole
experiments = mcc.get_experiments(injection_structure_ids=[frontal_pole_id])
print(f"\nNumber of experiments for Frontal pole: {len(experiments)}")

if len(experiments) > 0:
    exp = experiments[0]
    print(f"\nFirst experiment ID: {exp['id']}")
    
    # Try to get unionizes data
    unionizes = mcc.get_structure_unionizes([exp['id']], structure_ids=[frontal_pole_id])
    print(f"\nUnionizes data shape: {unionizes.shape}")
    print("\nUnionizes columns:", list(unionizes.columns))
    print("\nFirst unionize record:")
    print(unionizes.iloc[0])
    
    # Print the actual projection density values
    if 'projection_density' in unionizes.columns:
        print("\nProjection density values:")
        print(unionizes['projection_density'])
else:
    print("\nNo experiments found for Frontal pole")