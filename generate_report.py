import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def generate_html_report():
    print("Generating comprehensive report...")
    
    # HTML template start
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Region Connectivity Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Brain Region Connectivity Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Network Overview Section
    print("Adding network overview...")
    html += """
        <div class="section">
            <h2>1. Network Overview</h2>
    """
    
    try:
        matrix = pd.read_csv('network_metrics.csv')
        html += f"""
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Regions</td><td>{len(matrix)}</td></tr>
            </table>
        """
    except:
        html += "<p>Network metrics file not found.</p>"
    
    # Node Importance Section
    print("Adding node importance analysis...")
    html += """
        <div class="section">
            <h2>2. Node Importance Analysis</h2>
    """
    
    try:
        importance = pd.read_csv('node_importance.csv')
        top_nodes = importance.nlargest(10, 'betweenness')
        
        html += """
            <h3>Top 10 Most Important Nodes</h3>
            <table>
                <tr><th>Region</th><th>Betweenness Centrality</th></tr>
        """
        
        for _, row in top_nodes.iterrows():
            html += f"<tr><td>{row['region']}</td><td>{row['betweenness']:.4f}</td></tr>"
        
        html += "</table>"
    except:
        html += "<p>Node importance data not found.</p>"
    
    # Resilience Analysis Section
    print("Adding resilience analysis...")
    html += """
        <div class="section">
            <h2>3. Network Resilience Analysis</h2>
    """
    
    try:
        resilience = pd.read_csv('network_degradation.csv')
        html += """
            <h3>Network Degradation Summary</h3>
            <img src="network_degradation.png" alt="Network Degradation">
        """
    except:
        html += "<p>Resilience analysis data not found.</p>"
    
    # Expanded Node Removal Analysis
    print("Adding expanded node removal analysis...")
    html += """
        <div class="section">
            <h2>4. Expanded Node Removal Analysis</h2>
    """
    
    try:
        regional_impact = pd.read_csv('expanded_regional_impact.csv')
        html += """
            <h3>Regional Impact Distribution</h3>
            <img src="regional_impact_dist.png" alt="Regional Impact Distribution">
            
            <h3>Cascading Failure Analysis</h3>
            <img src="cascade_progression.png" alt="Cascading Failure Progression">
            
            <h3>Targeted vs Random Removal</h3>
            <img src="targeted_vs_random.png" alt="Targeted vs Random Comparison">
        """
    except:
        html += "<p>Expanded node removal analysis data not found.</p>"
    
    # Interactive Visualizations Section
    html += """
        <div class="section">
            <h2>5. Interactive Visualizations</h2>
        <p>The following interactive visualizations are available:</p>
        <ul>
            <li><a href="degradation_analysis.html">Interactive Network Degradation Analysis</a></li>
            <li><a href="resilience_analysis.html">Interactive Resilience Analysis</a></li>
        </ul>
    """
    
    # Conclusions Section
    html += """
        <div class="section">
            <h2>6. Key Findings</h2>
            <ul>
    """
    
    try:
        # Add key findings based on available data
        importance = pd.read_csv('node_importance.csv')
        top_node = importance.nlargest(1, 'betweenness').iloc[0]
        html += f"<li>Most critical region: {top_node['region']}</li>"
        
        regional_impact = pd.read_csv('expanded_regional_impact.csv')
        avg_impact = regional_impact['efficiency_change'].mean()
        html += f"<li>Average impact of node removal: {avg_impact:.2%} efficiency change</li>"
    except:
        html += "<li>Detailed findings not available</li>"
    
    html += """
            </ul>
        </div>
    """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Save report
    with open('brain_connectivity_report.html', 'w') as f:
        f.write(html)
    
    print("Report generated: brain_connectivity_report.html")

if __name__ == "__main__":
    generate_html_report()