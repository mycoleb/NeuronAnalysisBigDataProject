
# Neural Network Connectivity Analysis Project

## Key info for anyone trying to use this!
You need to be in the (brain_connectivity) environment for this to work.

## Running the project in bash
First, setup and verify the data:

Run setup_cache.py to initialize the Allen Brain Atlas connection

Run test_connection.py to verify the SDK is working

Run verify_data.py to check your database and connectivity matrix


Main analysis pipeline:

Run fetch_sql.py to get the latest connectivity data
Run analyze_patterns.py to get a basic overview of connectivity patterns
Run advanced_analysis.py for detailed network metrics and small-world properties


Visualizations and results:

Run seaborn_visualize.py for comprehensive static and interactive visualizations
Run generate_report.py to create a final HTML report summarizing all findings
## Executive Summary
This project analyzes how different regions of the mouse brain are connected to each other using data from the Allen Brain Atlas. Understanding these connections is crucial for neuroscience research as it helps reveal:
- How information flows through the brain
- Which regions are most important for brain function
- How resilient the brain's network is to damage
- How brain regions work together in communities

Through advanced network analysis techniques, we can identify critical brain regions, understand connection patterns, and visualize the complex web of neural connections in an interactive way.

[Add example visualization here]

## Key Findings
Our analysis reveals:
- The most strongly connected brain regions
- Critical hubs that connect different brain areas
- How the network degrades when connections are removed
- Communities of brain regions that work closely together

## Overview
This project provides a comprehensive framework for analyzing brain region connectivity patterns using data from the Allen Mouse Brain Connectivity Atlas. By leveraging advanced network science techniques, this toolkit enables researchers to explore and understand the complex interconnections between different brain regions, identify critical connectivity hubs, and analyze network resilience.

## Key Features

### Data Processing & Management
- Automated data extraction from Allen Brain Atlas API
- SQL database implementation for efficient storage
- Data validation and verification tools
- Connectivity matrix generation and processing

### Network Analysis Capabilities 
- Centrality measures (degree, betweenness, eigenvector, PageRank)
- Small-world property analysis
- Community detection and hierarchical organization
- Rich-club coefficient calculation
- Network resilience testing through node removal
- Cascading failure analysis

### Visualization Suite
- Interactive network graphs
- Dynamic heatmaps
- Connection strength distributions
- Hierarchical clustering dendrograms
- Network degradation plots
- Regional connectivity profiles

## Installation

### Environment Setup
```bash
# Create and activate conda environment
conda create -n brain_connectivity python=3.8
conda activate brain_connectivity

# Install required packages
pip install allensdk networkx numpy pandas seaborn plotly scipy statsmodels
Additional Requirements

SQLite3 for database management
Sufficient storage space for Allen Brain Atlas data (~2GB)
Modern web browser for interactive visualizations

Project Structure
Core Analysis Modules

a.py: Core connectivity analyzer class

Network initialization
Basic metric calculations
Graph construction and manipulation


advanced_analysis.py: Extended network analysis

Advanced centrality measures
Community detection
Path length analysis
Statistical testing


expanded_node_analysis.py: Node impact analysis

Individual node removal effects
Network degradation tracking
Criticality assessment



Data Management

fetch_sql.py: Data acquisition and storage

Allen API interaction
SQL database management
Data preprocessing
Matrix generation


verify_data.py: Data validation suite

Database integrity checks
Connectivity verification
Matrix validation



Visualization Tools

seaborn_visualize.py: Comprehensive visualization toolkit

Static plots generation
Interactive network graphs
Custom plotting functions
Multi-view data representation



Analysis Tools

resilience_analysis.py: Network stability assessment

Targeted node removal
Random failure simulation
Cascade effects
Recovery analysis



Usage Guide
Initial Setup

Data Acquisition:

bashCopypython fetch_sql.py
This initializes the database and downloads necessary data from Allen Brain Atlas.

Data Verification:

bashCopypython verify_data.py
Ensures data integrity and proper storage.
Running Analyses

Basic Network Analysis:

bashCopypython advanced_analysis.py
Generates core network metrics and basic visualizations.

Node Impact Analysis:

bashCopypython expanded_node_analysis.py
Analyzes individual node contributions and network resilience.

Visualization Generation:

bashCopypython seaborn_visualize.py
Creates comprehensive set of visualizations.
Viewing Results
Interactive Visualizations

Open interactive_network.html in a web browser for dynamic network exploration
View interactive_heatmap.html for detailed connectivity patterns
Use degradation_analysis.html for resilience visualization

Static Outputs

Network metrics: network_metrics.csv
Node importance: node_importance.csv
Community structure: community_structure.csv
Various PNG visualizations

Output Files Description
Data Files

connectivity_matrix.npy: Raw connectivity data
region_names.txt: Brain region identifiers
brain_connectivity.db: SQLite database

Analysis Results

network_metrics.csv: Basic network statistics
node_importance.csv: Node centrality measures
expanded_regional_impact.csv: Node removal effects
cascade_analysis.csv: Cascade failure results

Visualizations

connectivity_analysis.png: Overall network structure
network_degradation.png: Resilience visualization
hierarchical_organization.png: Community structure
Multiple interactive HTML files

Methodological Details
Network Construction

Directed graph representation
Weight-based edge creation
Custom thresholding

Analysis Metrics

Multiple centrality measures
Path length calculations
Community detection algorithms
Resilience metrics

Statistical Analysis

Significance testing
Random network comparisons
Distribution analysis

Contributing
Contributions are welcome! Please read the contribution guidelines before submitting pull requests.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Allen Institute for Brain Science
NetworkX developers
Scientific Python community

Citation
If you use this toolkit in your research, please cite:
Copy@software{brain_connectivity_toolkit,
  title={Neural Network Connectivity Analysis Project},
  author={[Your Name]},
  year={2024},
  url={https://github.com/mycoleb/NeuronAnalysisBigDataProject}
}
Contact
For questions and support, please open an issue on the GitHub repository.# Allen Brain Atlas Connectivity Measurement

#Additional Info about my methodology 
In the Allen Brain Atlas methodology, "proportion of signal pixels" refers to the quantification of neural connections using fluorescent imaging. Here's what it specifically means:

## Measurement Process
1. When a tracer is injected into a source region, it travels along the axons to connected regions
2. The tracer fluoresces (glows) when imaged
3. Each brain region is imaged and divided into pixels
4. The "signal pixels" are pixels that show fluorescence above a threshold (indicating presence of traced connections)

## Calculation
The proportion is calculated as:

```math
Projection Density = \frac{Number\;of\;signal\;pixels}{Total\;pixels\;in\;target\;region}
