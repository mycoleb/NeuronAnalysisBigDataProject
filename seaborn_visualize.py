import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from scipy.cluster import hierarchy

# Verify plotly import
try:
    import plotly
    import plotly.graph_objects as go
    print("Successfully imported plotly version:", plotly.__version__)
except ImportError:
    print("Error: plotly is not installed. Please install it using:")
    print("pip install plotly")
    exit(1)

# Function to visualize connectivity


def visualize_connectivity():
    # Load data
    matrix = np.load('connectivity_matrix.npy')
    with open('region_names.txt', 'r') as f:
        regions = [line.strip() for line in f]
    
    # Create figure and axes with specific size
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    density_label = 'Projection Density\n(fraction of region containing traced connections)'
    
    # 1. Connectivity Heatmap
    heatmap = sns.heatmap(matrix, cmap='Blues', xticklabels=False, yticklabels=False,
                         ax=axes[0,0], cbar_kws={'label': density_label})
    axes[0,0].set_title('Full Connectivity Matrix')
    axes[0,0].set_xlabel('Target Region Index')
    axes[0,0].set_ylabel('Source Region Index')
    
    # 2. Top 10 Strongest Connections
    flat_indices = np.argsort(matrix.flatten())[-10:]
    row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)
    
    strongest_connections = pd.DataFrame({
        'Source': [regions[i] for i in row_indices],
        'Target': [regions[i] for i in col_indices],
        'Strength': matrix[row_indices, col_indices]
    })
    
    sns.barplot(data=strongest_connections, x='Strength', y='Source', ax=axes[0,1])
    axes[0,1].set_title('Top 10 Strongest Connections')
    axes[0,1].set_xlabel(density_label)
    axes[0,1].set_ylabel('Source Region')
    
    # 3. Region Connection Counts
    out_degree = np.sum(matrix > 0, axis=1)
    in_degree = np.sum(matrix > 0, axis=0)
    
    connection_counts = pd.DataFrame({
        'Region': regions,
        'Outgoing': out_degree,
        'Incoming': in_degree
    })
    
    top_connected = connection_counts.nlargest(10, 'Outgoing')
    
    sns.barplot(data=pd.melt(top_connected, id_vars=['Region'], 
                            value_vars=['Outgoing', 'Incoming']),
                x='value', y='Region', hue='variable', ax=axes[1,0])
    axes[1,0].set_title('Top 10 Most Connected Regions')
    axes[1,0].set_xlabel('Number of Connections (count)')
    axes[1,0].set_ylabel('Brain Region')
    
    # 4. Connection Strength Distribution (excluding zeros)
    nonzero_connections = matrix[matrix > 0].flatten()
    sns.histplot(nonzero_connections, bins=50, ax=axes[1,1])
    axes[1,1].set_title('Distribution of Non-Zero Connection Strengths')
    axes[1,1].set_xlabel(density_label)
    axes[1,1].set_ylabel('Frequency (count)')
    
    plt.tight_layout()
    plt.savefig('connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed heatmap of top regions
    plt.figure(figsize=(15, 15))
    top_indices = connection_counts.nlargest(20, 'Outgoing').index
    top_matrix = matrix[top_indices][:, top_indices]
    top_names = [regions[i] for i in top_indices]
    
    detailed_heatmap = sns.heatmap(top_matrix, cmap='Blues', 
                                  xticklabels=top_names, yticklabels=top_names,
                                  annot=True, fmt='.2f', square=True,
                                  cbar_kws={'label': density_label})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Connectivity Matrix for Top 20 Most Connected Regions')
    plt.xlabel('Target Region')
    plt.ylabel('Source Region')
    
    plt.tight_layout()
    plt.savefig('top_regions_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    return connection_counts

def create_additional_plots(matrix, regions, connection_counts):
    # Network graph of strongest connections
    plt.figure(figsize=(15, 15))
    G = nx.Graph()
    threshold = np.percentile(matrix[matrix > 0], 95)  # Top 5% of connections
    for i in range(len(regions)):
        for j in range(len(regions)):
            if matrix[i,j] > threshold:
                G.add_edge(regions[i], regions[j], weight=matrix[i,j])
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=100, node_color='lightblue',
            with_labels=True, font_size=8)
    plt.title('Network Graph of Strongest Connections')
    plt.savefig('network_graph.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Regional Connectivity Profile
    plt.figure(figsize=(15, 8))
    top_regions = connection_counts.nlargest(10, 'Outgoing')
    sns.scatterplot(data=connection_counts, 
                   x='Outgoing', y='Incoming', alpha=0.5)
    plt.title('Regional Connectivity Profile')
    plt.xlabel('Number of Outgoing Connections')
    plt.ylabel('Number of Incoming Connections')
    
    # Label top regions
    for idx, row in top_regions.iterrows():
        plt.annotate(row['Region'], 
                    (row['Outgoing'], row['Incoming']),
                    xytext=(5, 5), textcoords='offset points')
    plt.savefig('connectivity_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Reciprocal Connections Analysis
    reciprocal_mask = np.logical_and(matrix > 0, matrix.T > 0)
    reciprocal_strengths = pd.DataFrame({
        'Forward': matrix[reciprocal_mask],
        'Reverse': matrix.T[reciprocal_mask]
    })
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=reciprocal_strengths, x='Forward', y='Reverse', alpha=0.5)
    plt.plot([0, max(matrix.max(), matrix.T.max())], 
             [0, max(matrix.max(), matrix.T.max())], 'r--')
    plt.title('Reciprocal Connections Analysis')
    plt.xlabel('Forward Connection Strength')
    plt.ylabel('Reverse Connection Strength')
    plt.savefig('reciprocal_connections.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Hierarchical clustering of regions
    plt.figure(figsize=(15, 10))
    linkage = hierarchy.linkage(matrix, method='ward')
    dendrogram = hierarchy.dendrogram(linkage, labels=regions)
    plt.title('Hierarchical Organization of Brain Regions')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('hierarchical_organization.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_network(matrix, regions):
    # Create network of strongest connections
    G = nx.Graph()
    threshold = np.percentile(matrix[matrix > 0], 95)  # Top 5% of connections
    
    # Add edges for strong connections
    for i in range(len(regions)):
        for j in range(len(regions)):
            if matrix[i,j] > threshold:
                G.add_edge(regions[i], regions[j], weight=matrix[i,j])
    
    # Use force-directed layout for node positions
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G.edges[edge]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Calculate node sizes based on degree
    node_degrees = [G.degree(node) for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[d * 2 for d in node_degrees],  # Size nodes by degree
            color=node_degrees,
            line_width=2))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Interactive Network of Brain Region Connections',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    # Save as HTML file
    fig.write_html("interactive_network.html")
    
    return fig

def create_interactive_heatmap(matrix, regions):
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=regions,
        y=regions,
        colorscale='Blues',
        hoverongaps=False))
    
    fig.update_layout(
        title='Interactive Brain Region Connectivity Heatmap',
        xaxis_title="Target Region",
        yaxis_title="Source Region",
        width=1200,
        height=1200
    )
    
    # Save as HTML file
    fig.write_html("interactive_heatmap.html")
    
    return fig

if __name__ == "__main__":
    # Load data and create base visualizations
    connection_counts = visualize_connectivity()
    
    # Load data again for additional plots
    matrix = np.load('connectivity_matrix.npy')
    with open('region_names.txt', 'r') as f:
        regions = [line.strip() for line in f]
    
    # Create additional visualizations
    create_additional_plots(matrix, regions, connection_counts)
    
    # Create interactive visualizations
    create_interactive_network(matrix, regions)
    create_interactive_heatmap(matrix, regions)
    
    print("All visualizations have been saved:"
          "\n- connectivity_analysis.png"
          "\n- top_regions_heatmap.png"
          "\n- network_graph.png"
          "\n- connectivity_profile.png"
          "\n- reciprocal_connections.png"
          "\n- hierarchical_organization.png"
          "\n- interactive_network.html"
          "\n- interactive_heatmap.html")