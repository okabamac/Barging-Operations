import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from PIL import Image  # Import PIL for image handling

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Title for the application
st.title("NNPC Command & Control Centre")

# Load the image for the sidebar icon and resize it
image = Image.open('NNPC-Logo.png')
image = image.resize((80, 50))  # Resize the image to 80x50 pixels for example
st.sidebar.image(image, use_column_width=False)  # Adjust use_column_width to False for smaller images

st.write("This app shows the number of AIS infractions for each item in the shuttle fleet.", unsafe_allow_html=True)

st.sidebar.write("Barging Operations")

# Read the CSV file
csv_file = 'shuttles.csv'
df = pd.read_csv(csv_file)

# Calculate total AIS infractions for each item
df_totals = df.groupby('Item')['AIS_Infraction'].sum().reset_index()

# Merge total AIS infractions back into the main DataFrame
df = pd.merge(df, df_totals, on='Item', suffixes=('', '_total'))

# Sidebar menu for selecting item
selected_item = st.sidebar.selectbox('Select Asset', df['Item'].unique())

# Filter DataFrame based on selected item
df_selected = df[df['Item'] == selected_item].copy()  # Make a copy to avoid SettingWithCopyWarning

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
FSO = df_selected.iloc[0]['FSO']  # Assuming FSO is the same for all shuttles under the selected item
G.add_node(FSO, type='FSO')

# Add edges: shuttle -> FSO
for shuttle in df_selected['Shuttle'].unique():
    G.add_edge(shuttle, FSO)

# Custom layout positioning
pos = {}
level_height = 100  # Height between levels
node_distance = 50  # Horizontal distance between nodes

# Position items at the top level
for i, item in enumerate(df_selected['Item'].unique()):
    pos[item] = (i * node_distance, level_height * 2)

# Position shuttles and jetties in the middle level
if has_jetty:
    for i, jetty in enumerate(df_selected['Jetty'].dropna().unique()):
        pos[jetty] = (i * node_distance, level_height)
    for i, shuttle in enumerate(df_selected['Shuttle'].unique()):
        pos[shuttle] = (i * node_distance, level_height * 0.5)  # Adjust y-position to differentiate from jetties
else:
    for i, shuttle in enumerate(df_selected['Shuttle'].unique()):
        pos[shuttle] = (i * node_distance, level_height)

# Position FSO at the bottom level
pos[FSO] = ((len(df_selected['Shuttle'].unique()) - 1) * node_distance / 2, 0)

# Determine color scale based on AIS Infractions
max_aif = max(df_selected['AIS_Infraction'])  # Maximum AIS Infraction among shuttles
min_aif = min(df_selected['AIS_Infraction'])  # Minimum AIS Infraction among shuttles

# Function to map AIS Infractions to color scale (red to green)
def calculate_color(aif):
    # Handle case where all AIS Infraction values are zero
    if min_aif == max_aif:
        return 'rgb(0, 255, 0)'  # Return green color for zero AIS Infractions

    normalized_aif = (aif - min_aif) / (max_aif - min_aif)  # Normalize AIS Infractions to range [0, 1]
    r = int(255 * normalized_aif)  # Red component increases from 0 (green) to 255 (red)
    g = int(255 * (1 - normalized_aif))  # Green component decreases from 255 (red) to 0 (green)
    return f'rgb({r}, {g}, 0)'  # RGB color format

# Create edge traces
edge_trace = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    if edge[0] == FSO or edge[1] == FSO:
        line_color = 'blue'  # Color for FSO edges
    else:
        line_color = 'gray'  # Color for other edges
    edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines', line=dict(width=2, color=line_color), hoverinfo='none'))

# Create node traces with custom color scale
node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', hoverinfo='text', textposition='top center', marker=dict(size=20, color=[], showscale=False))

for node in G.nodes():
    x, y = pos[node]
    label = G.nodes[node].get('label', node)  # Get node label including AIS Infractions
    node_trace['x'] += (x,)
    node_trace['y'] += (y,)
    node_trace['text'] += (label,)  # Use label instead of node for text display
    
    # Color nodes based on AIS Infraction value
    if G.nodes[node]['type'] == 'Shuttle':
        node_trace['marker']['color'] += (calculate_color(G.nodes[node]['AIS_Infraction']),)  # Use custom function to determine color based on AIS Infraction value
    else:
        node_trace['marker']['color'] += ('orange',)  # Default color for non-shuttle nodes

# Create figure
fig = go.Figure(data=edge_trace + [node_trace],
                layout=go.Layout(
                    title='Decomposition Tree',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

# Display the graph in Streamlit
st.plotly_chart(fig, use_container_width=True)
