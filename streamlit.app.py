import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from PIL import Image

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Custom CSS to maximize available space
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

# Title for the application
st.title("NNPC Command & Control Centre")

# Load the image for the sidebar icon and resize it
try:
    image = Image.open('NNPC-Logo.png')
    image = image.resize((80, 50))
    st.sidebar.image(image, use_container_width=False)
except Exception as e:
    st.sidebar.warning("Could not load logo image. Please ensure 'NNPC-Logo.png' is in the correct directory.")

st.write("This app shows the barging route from asset to export.", unsafe_allow_html=True)

st.sidebar.write("Barging Operations")

# Read the CSV file
try:
    csv_file = 'shuttles.csv'
    df = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Calculate total AIS infractions for each item
df_totals = df.groupby('Item')['AIS_Infraction'].sum().reset_index()

# Merge total AIS infractions back into the main DataFrame
df = pd.merge(df, df_totals, on='Item', suffixes=('', '_total'))

# Sidebar menu for selecting item
selected_item = st.sidebar.selectbox('Select Asset', df['Item'].unique())

# Filter DataFrame based on selected item
df_selected = df[df['Item'] == selected_item].copy()

# Create a directed graph
G = nx.DiGraph()

# Add nodes for items
for item in df_selected['Item'].unique():
    G.add_node(item, type='Item')

# Check if the Jetty column exists in the selected item's data
if 'Jetty' in df_selected.columns:
    has_jetty = True
    # Add nodes for jetties
    for jetty in df_selected['Jetty'].dropna().unique():
        G.add_node(jetty, type='Jetty')

    # Add edges: item -> jetty -> shuttle
    for index, row in df_selected.iterrows():
        if pd.notna(row['Jetty']):
            G.add_edge(row['Item'], row['Jetty'])
            G.add_edge(row['Jetty'], row['Shuttle'], AIS_Infraction=row['AIS_Infraction'])
        else:
            G.add_edge(row['Item'], row['Shuttle'], AIS_Infraction=row['AIS_Infraction'])
else:
    has_jetty = False
    # Add edges: item -> shuttle
    for index, row in df_selected.iterrows():
        G.add_edge(row['Item'], row['Shuttle'], AIS_Infraction=row['AIS_Infraction'])

# Add nodes for shuttles with AIS Infractions details
for index, row in df_selected.iterrows():
    shuttle_label = f"{row['Shuttle']} ({row['AIS_Infraction']} AIS Infraction{'s' if row['AIS_Infraction'] > 1 else ''})"
    G.add_node(row['Shuttle'], type='Shuttle', label=shuttle_label, AIS_Infraction=row['AIS_Infraction'])

# Add node for FSO
FSO = df_selected.iloc[0]['FSO']
G.add_node(FSO, type='FSO')

# Add edges: shuttle -> FSO
for shuttle in df_selected['Shuttle'].unique():
    G.add_edge(shuttle, FSO)

# Custom layout positioning
pos = {}
level_height = 200  # Increased from 100 to 200
node_distance = 50

# Position items at the top level
for i, item in enumerate(df_selected['Item'].unique()):
    pos[item] = (i * node_distance, level_height * 3)  # Increased multiplier from 2 to 3

# Position shuttles and jetties in the middle level
if has_jetty:
    for i, jetty in enumerate(df_selected['Jetty'].dropna().unique()):
        pos[jetty] = (i * node_distance, level_height * 2)  # Increased multiplier from 1 to 2
    for i, shuttle in enumerate(df_selected['Shuttle'].unique()):
        pos[shuttle] = (i * node_distance, level_height)
else:
    for i, shuttle in enumerate(df_selected['Shuttle'].unique()):
        pos[shuttle] = (i * node_distance, level_height * 1.5)  # Increased multiplier from 1 to 1.5

# Position FSO at the bottom level
pos[FSO] = ((len(df_selected['Shuttle'].unique()) - 1) * node_distance / 2, 0)

# Determine color scale based on AIS Infractions
max_aif = max(df_selected['AIS_Infraction'])
min_aif = min(df_selected['AIS_Infraction'])

# Function to map AIS Infractions to color scale (red to green)
def calculate_color(aif):
    if min_aif == max_aif:
        return 'rgb(0, 255, 0)'
    normalized_aif = (aif - min_aif) / (max_aif - min_aif)
    r = int(255 * normalized_aif)
    g = int(255 * (1 - normalized_aif))
    return f'rgb({r}, {g}, 0)'

# Create edge traces
edge_trace = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    if edge[0] == FSO or edge[1] == FSO:
        line_color = 'blue'
    else:
        line_color = 'gray'
    edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=2, color=line_color), hoverinfo='none'))

# Create node traces with custom color scale
node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', hoverinfo='text', textposition='top center', marker=dict(size=20, color=[], showscale=False))

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

# Create figure with fixed layout
fig = go.Figure()

# Add all the edge traces
for trace in edge_trace:
    fig.add_trace(trace)

# Add the node trace
fig.add_trace(node_trace)

# Update layout - FIXED version
fig.update_layout(
    title=dict(
        text='Barging Ops',
        font=dict(size=16)
    ),
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=800
    # width removed or fixed earlier
)


# Create a container for the graph
graph_container = st.container()

# Inside the container, display the graph with the proper parameter
with graph_container:
    st.plotly_chart(fig, use_container_width=True)

# Add a simple data view below the graph
st.subheader(f"AIS Data Data for {selected_item}")
st.dataframe(df_selected[['Shuttle', 'AIS_Infraction', 'FSO']], use_container_width=True)