from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

# Initialize cache and verify it works
mcc = MouseConnectivityCache(manifest_file='mouse_connectivity_manifest.json')
print("Cache initialized successfully. Manifest file created.")
structure_tree = mcc.get_structure_tree()
print(f"Connected to Allen Brain Atlas. Ready to fetch data.")