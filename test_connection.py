from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import allensdk

# Print the version to help with debugging
print(f"AllenSDK version: {allensdk.__version__}")

# Initialize the cache
mcc = MouseConnectivityCache(manifest_file='mouse_connectivity_manifest.json')
structure_tree = mcc.get_structure_tree()

# Test a simple query
structures = structure_tree.get_structures_by_set_id([167587189])
print(f"\nNumber of structures found: {len(structures)}")
print(f"\nFirst structure: {structures[0] if structures else 'None'}")

# Print the structure names if any were found
if structures:
    print("\nAll structure names:")
    for structure in structures:
        print(f"- {structure['name']}")