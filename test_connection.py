# Import the MouseConnectivityCache class from the AllenSDK.
# This class provides an interface to download and locally cache
# mouse brain connectivity data from the Allen Institute.
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

# Import the base AllenSDK package so we can access metadata such as the version.
import allensdk


# ------------------------------------------------------------
# Print the AllenSDK version
# ------------------------------------------------------------
# This is useful for debugging and reproducibility.
# If someone else runs this script and gets different results,
# knowing the SDK version can help identify compatibility issues.
print(f"AllenSDK version: {allensdk.__version__}")


# ------------------------------------------------------------
# Initialize the Mouse Connectivity Cache
# ------------------------------------------------------------
# The manifest file stores information about what data has already
# been downloaded and where it is stored locally.
#
# If the manifest file does not exist yet, AllenSDK will create it.
# This helps avoid re-downloading large neuroscience datasets.
mcc = MouseConnectivityCache(manifest_file='mouse_connectivity_manifest.json')


# ------------------------------------------------------------
# Retrieve the brain structure hierarchy
# ------------------------------------------------------------
# The structure tree is a hierarchical representation of brain regions
# used by the Allen Brain Atlas.
#
# Each structure contains metadata such as:
# - structure name
# - acronym
# - parent region
# - ontology ID
structure_tree = mcc.get_structure_tree()


# ------------------------------------------------------------
# Query brain structures by structure set ID
# ------------------------------------------------------------
# Structure sets group related brain regions together.
#
# The ID 167587189 corresponds to a predefined structure set
# in the Allen Brain Atlas ontology.
#
# This query returns a list of dictionaries, where each dictionary
# represents a brain structure.
structures = structure_tree.get_structures_by_set_id([167587189])


# ------------------------------------------------------------
# Print basic information about the query results
# ------------------------------------------------------------

# Print how many brain structures were found in the selected set.
print(f"\nNumber of structures found: {len(structures)}")

# Print the first structure as an example to inspect its fields.
# If the list is empty, print "None".
print(f"\nFirst structure: {structures[0] if structures else 'None'}")


# ------------------------------------------------------------
# Print all structure names
# ------------------------------------------------------------
# If the query returned any results, iterate through them
# and print the name of each brain region.
#
# Each structure dictionary contains several keys like:
# - 'name'
# - 'acronym'
# - 'id'
# - 'parent_structure_id'
if structures:
    print("\nAll structure names:")

    # Loop through each structure in the returned list
    for structure in structures:

        # Access the structure name using the 'name' key
        # and print it in a readable format
        print(f"- {structure['name']}")
