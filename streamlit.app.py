import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding: 2rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("NNPC Command & Control Centre")

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

# Totals
df_totals = df.groupby('Item')['AIS_Infraction'].sum().reset_index()
df = pd.merge(df, df_totals, on='Item', suffixes=('', '_total'))

# Sidebar selector
selected_item = st.sidebar.radio('Select Asset', df['Item'].unique())

df_selected = df[df['Item'] == selected_item].copy()

# Graph init
G = nx.DiGraph()
for item in df_selected['Item'].unique():
    G.add_node(item, type='Item')

FSO = df_selected.iloc[0]['FSO']
G.add_node(FSO, type='FSO')

has_jetty = 'Jetty' in df_selected.columns
if has_jetty:
    for jetty in df_selected['Jetty'].dropna().unique():
        G.add_node(jetty, type='Jetty')

# Add nodes & edges
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

# Layout positions
pos = {}
level_height = 200
node_distance = 50

for i, item in enumerate(df_selected['Item'].unique()):
    pos[item] = (i * node_distance, level_height * 3)

if has_jetty:
    for i, jetty in enumerate(df_selected['Jetty'].dropna().unique()):
        pos[jetty] = (i * node_distance, level_height * 2)
    for i, shuttle in enumerate(df_selected['Shuttle'].dropna().unique()):
        pos[shuttle] = (i * node_distance, level_height)
else:
    for i, shuttle in enumerate(df_selected['Shuttle'].dropna().unique()):
        pos[shuttle] = (i * node_distance, level_height * 1.5)

pos[FSO] = ((len(df_selected['Shuttle'].dropna().unique()) - 1) * node_distance / 2, 0)

# Color mapping
max_aif = max(df_selected['AIS_Infraction']) if not df_selected.empty else 1
min_aif = min(df_selected['AIS_Infraction']) if not df_selected.empty else 0

def calculate_color(aif):
    if max_aif == min_aif:
        return 'rgb(0, 255, 0)'
    norm = (aif - min_aif) / (max_aif - min_aif)
    r = int(255 * norm)
    g = int(255 * (1 - norm))
    return f'rgb({r}, {g}, 0)'

# Edge trace
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

# Node trace (text only, no circle markers)
node_trace = go.Scatter(
    x=[], y=[], text=[], mode='text',
    hoverinfo='text', textfont=dict(size=17),  # Increased font size
    textposition='top center'
)

for node in G.nodes():
    x, y = pos[node]
    label = G.nodes[node].get('label', node)
    node_trace['x'] += (x,)
    node_trace['y'] += (y,)
    node_trace['text'] += (label,)

# Image overlay logic
images = []

def resolve_image(node_type):
    base = f"barge_images/{node_type.lower()}.png"
    fallback = "barge_images/default.png"
    return base if os.path.exists(base) else fallback

def add_image_node(img_path, x, y):
    return dict(
        source=img_path,
        xref="x", yref="y",
        x=x - 10,
        y=y + 10,
        sizex=20, sizey=20,
        xanchor="left", yanchor="bottom",
        layer="above"
    )

for node, attrs in G.nodes(data=True):
    x, y = pos[node]
    node_type = attrs.get("type")
    img_path = resolve_image(node_type)
    images.append(add_image_node(img_path, x, y))

# Final figure
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
    height=800,
    images=images
)

# Display
st.plotly_chart(fig, use_container_width=True)
st.subheader(f"AIS Data for {selected_item}")
st.dataframe(df_selected[['Shuttle', 'AIS_Infraction', 'FSO']], use_container_width=True)
