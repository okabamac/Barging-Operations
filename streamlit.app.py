import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from PIL import Image

# Set page config
st.set_page_config(layout="wide")

# Custom CSS for Modern UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide top header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #00d2ff;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background: rgba(28, 35, 49, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("NNPC Command & Control Centre")

import os

# Sidebar image
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, 'NNPC-Logo.png')
    image = Image.open(logo_path)
    image = image.resize((80, 50))
    st.sidebar.image(image, use_container_width=False)
except Exception as e:
    st.sidebar.warning(f"Logo not found. Ensure 'NNPC-Logo.png' is available. Error: {e}")

st.markdown("<p style='font-size: 1.1rem; color: #a9b5c7;'>Monitor, track, and visualize the real-time barging route network from asset to export terminals.</p>", unsafe_allow_html=True)
st.sidebar.markdown("### Control Panel")

# Read data
try:
    csv_path = os.path.join(script_dir, 'shuttles.csv')
    df = pd.read_csv(csv_path)
    df['AIS_Infraction'] = df['AIS_Infraction'].fillna(0).astype(int)
except Exception as e:
    st.error(f"Error loading CSV from {csv_path}: {e}")
    st.stop()

# Calculate totals
df_totals = df.groupby('Item')['AIS_Infraction'].sum().reset_index()
df = pd.merge(df, df_totals, on='Item', suffixes=('', '_total'))

# Sidebar radio menu
selected_item = st.sidebar.radio('Select Asset', df['Item'].unique())

# Filter by selection
df_selected = df[df['Item'] == selected_item].copy()

# Dashboard Metrics Layer
st.markdown("### Command Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Shuttles Active", len(df_selected['Shuttle'].dropna().unique()))
with col2:
    st.metric("Total AIS Infractions", int(df_selected['AIS_Infraction'].sum()))
with col3:
    st.metric("Total Jetties", len(df_selected['Jetty'].dropna().unique()) if 'Jetty' in df_selected.columns else 0)

st.divider()

# Directed graph
G = nx.DiGraph()

# Add item node
for item in df_selected['Item'].unique():
    G.add_node(item, type='Item')

# Add FSO node
FSO = df_selected.iloc[0]['FSO']
G.add_node(FSO, type='FSO')

# Add Jetty nodes
has_jetty = 'Jetty' in df_selected.columns
if has_jetty:
    for jetty in df_selected['Jetty'].dropna().unique():
        G.add_node(jetty, type='Jetty')

# Add shuttle nodes and edges
for _, row in df_selected.iterrows():
    item = row['Item']
    jetty = row['Jetty'] if 'Jetty' in row and pd.notna(row['Jetty']) else None
    shuttle = row['Shuttle'] if 'Shuttle' in row and pd.notna(row['Shuttle']) else None
    aif = row['AIS_Infraction']
    imo = row.get('Shuttle IMO', 'N/A')
    cap = row.get('Shuttle Capacity', 'N/A')
    status = row.get('Shuttle Status', 'Unknown')

    if shuttle:
        status_emoji = '✅' if status == 'Active' else ('❌' if status == 'Inactive' else '➖')
        label = f"{status_emoji} {shuttle} ({imo})"
        hover = f"<b>{shuttle}</b><br>IMO: {imo}<br>Capacity: {cap}<br>AIS Count: {aif}<br>Status: {status}"
        G.add_node(shuttle, type='Shuttle', label=label, hover=hover, AIS_Infraction=aif)

        if jetty:
            G.add_edge(item, jetty)
            G.add_edge(jetty, shuttle, AIS_Infraction=aif)
            G.add_edge(shuttle, FSO)
        else:
            G.add_edge(item, shuttle, AIS_Infraction=aif)
            G.add_edge(shuttle, FSO)
    else:
        if jetty:
            G.add_edge(item, jetty)
            G.add_edge(jetty, FSO)
        else:
            G.add_edge(item, FSO)

# Node layout
pos = {}
level_height = 200
node_distance = 50

# Top level: Item
for i, item in enumerate(df_selected['Item'].unique()):
    pos[item] = (i * node_distance, level_height * 3)

# Middle: Jetty and Shuttle
if has_jetty:
    for i, jetty in enumerate(df_selected['Jetty'].dropna().unique()):
        pos[jetty] = (i * node_distance, level_height * 2)
    for i, shuttle in enumerate(df_selected['Shuttle'].dropna().unique()):
        pos[shuttle] = (i * node_distance, level_height)
else:
    for i, shuttle in enumerate(df_selected['Shuttle'].dropna().unique()):
        pos[shuttle] = (i * node_distance, level_height * 1.5)

# Bottom: FSO
pos[FSO] = ((len(df_selected['Shuttle'].dropna().unique()) - 1) * node_distance / 2, 0)

# Color scale
max_aif = max(df_selected['AIS_Infraction']) if not df_selected.empty else 1
min_aif = min(df_selected['AIS_Infraction']) if not df_selected.empty else 0

def calculate_color(aif):
    if max_aif == min_aif:
        return 'rgb(0, 255, 0)'
    norm = (aif - min_aif) / (max_aif - min_aif)
    r = int(255 * norm)
    g = int(255 * (1 - norm))
    return f'rgb({r}, {g}, 0)'

# Edge traces
edge_trace = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    color = '#00d2ff' if FSO in edge else 'rgba(255, 255, 255, 0.2)'
    edge_trace.append(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode='lines',
        line=dict(width=2, color=color),
        hoverinfo='none'
    ))

# Node trace
node_trace = go.Scatter(
    x=[], y=[], text=[], mode='markers+text',
    hoverinfo='text', hovertext=[], textposition='top center',
    marker=dict(size=24, color=[], line=dict(width=2, color='white'), showscale=False)
)

for node in G.nodes():
    x, y = pos[node]
    label = G.nodes[node].get('label', node)
    hover_info = G.nodes[node].get('hover', label)
    node_trace['x'] += (x,)
    node_trace['y'] += (y,)
    node_trace['text'] += (label,)
    node_trace['hovertext'] += (hover_info,)
    
    ntype = G.nodes[node]['type']
    if ntype == 'Shuttle':
        node_trace['marker']['color'] += (calculate_color(G.nodes[node]['AIS_Infraction']),)
    elif ntype == 'Item':
        node_trace['marker']['color'] += ('#00d2ff',)
    elif ntype == 'FSO':
        node_trace['marker']['color'] += ('#00ff9d',)
    else:
        node_trace['marker']['color'] += ('#ffb703',)

# Plotly figure
fig = go.Figure()
for trace in edge_trace:
    fig.add_trace(trace)
fig.add_trace(node_trace)

fig.update_layout(
    title=dict(text='Interactive Barging Routes Map', font=dict(size=22, color='#ffffff', family="Inter, sans-serif")),
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=20, r=20, t=60),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=750,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

# Show chart
st.plotly_chart(fig, use_container_width=True)

# Data table
st.markdown("### Operational Data Records")
cols_to_show = ['Shuttle', 'Shuttle IMO', 'Shuttle Capacity', 'Shuttle Status', 'AIS_Infraction', 'FSO']
cols_to_show = [c for c in cols_to_show if c in df_selected.columns]
st.dataframe(
    df_selected[cols_to_show],
    use_container_width=True,
    hide_index=True
)
