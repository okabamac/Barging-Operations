import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from PIL import Image

# Set page config
st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("NNPC Command & Control Centre")

# Sidebar image
try:
    image = Image.open('NNPC-Logo.png')
    image = image.resize((80, 50))
    st.sidebar.image(image, use_container_width=False)
except Exception:
    st.sidebar.warning("Logo not found. Ensure 'NNPC-Logo.png' is available.")

st.write("This app shows the barging route from asset to export.")
st.sidebar.write("Barging Operations")

# Read data
try:
    df = pd.read_csv('shuttles.csv')
    df['AIS_Infraction'] = df['AIS_Infraction'].fillna(0).astype(int)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Calculate totals
df_totals = df.groupby('Item')['AIS_Infraction'].sum().reset_index()
df = pd.merge(df, df_totals, on='Item', suffixes=('', '_total'))

# Sidebar radio menu
selected_item = st.sidebar.radio('Select Asset', df['Item'].unique())

# Filter by selection
df_selected = df[df['Item'] == selected_item].copy()

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

    if shuttle:
        label = f"{shuttle} ({aif})"
        G.add_node(shuttle, type='Shuttle', label=label, AIS_Infraction=aif)

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
    color = 'blue' if FSO in edge else 'gray'
    edge_trace.append(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode='lines',
        line=dict(width=2, color=color),
        hoverinfo='none'
    ))

# Node trace
node_trace = go.Scatter(
    x=[], y=[], text=[], mode='markers+text',
    hoverinfo='text', textposition='top center',
    marker=dict(size=20, color=[], showscale=False)
)

for node in G.nodes():
    x, y = pos[node]
    label = G.nodes[node].get('label', node)
    node_trace['x'] += (x,)
    node_trace['y'] += (y,)
    node_trace['text'] += (label,)
    if G.nodes[node]['type'] == 'Shuttle':
        node_trace['marker']['color'] += (calculate_color(G.nodes[node]['AIS_Infraction']),)
    else:
        node_trace['marker']['color'] += ('orange',)

# Plotly figure
fig = go.Figure()
for trace in edge_trace:
    fig.add_trace(trace)
fig.add_trace(node_trace)

fig.update_layout(
    title=dict(text='Barging Ops', font=dict(size=16)),
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=800
)

# Show chart
st.plotly_chart(fig, use_container_width=True)

# Data table
st.subheader(f"AIS Data for {selected_item}")
st.dataframe(df_selected[['Shuttle', 'AIS_Infraction', 'FSO']], use_container_width=True)
