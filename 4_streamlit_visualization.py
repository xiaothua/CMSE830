import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 1. Set up Streamlit app with single page
st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

# Page: Project Overview
st.title("Exploring the Correlation between Electricity and Economy")
st.write("The main goal:")
st.write("1) Explore the correlation between global economic data and electricity data to identify the representative features.")
st.write("2) Examine differences in electricity strategies across countries at different economic stages.")

# Add a table to display the features of the two datasets
electricity_features = [
    'Electricity: Net Generation',
    'Electricity: Net Consumption',
    'Electricity: Imports',
    'Electricity: Exports',
    'Electricity: Net Imports',
    'Electricity: Installed Capacity',
    'Electricity: Distribution Losses'
]

economics_features = [
    'Economics: GDP Per Capita',
    'Economics: GDP',
    'Economics: GDP Growth',
    'Economics: PPP Per Capita',
    'Economics: PPP (Purchasing Power Parity)',
    'Economics: PPP Growth'
]

# Create a DataFrame to display the features
data = {
    "Electricity Features": electricity_features + [None] * (len(economics_features) - len(electricity_features)),
    "Economics Features": economics_features + [None] * (len(electricity_features) - len(economics_features))
}

features_df = pd.DataFrame(data)

# Display the features table
st.write("## Features of the Datasets")
st.dataframe(features_df, use_container_width=True)

# Load data_imputation from CSV file
data_imputation = pd.read_csv('data_imputation.csv', index_col=[0, 1])

# 3. Dividing the countries into 3 groups based on electricity import/export strategy
# 3.1 Calculate net import ratio
try:
    net_imports = data_imputation.xs('Electricity: Net Imports', level=1)
    net_generation = data_imputation.xs('Electricity: Net Generation', level=1).replace(0, np.nan)  # Avoid division by zero
    net_import_ratio = net_imports.div(net_generation)
except KeyError:
    st.error("The data_imputation DataFrame does not contain the necessary levels. Please check the input data.")

# 3.2 Function to plot the net import ratio on the world map for each year
def plot_world_map(df, title):
    fig = go.Figure()

    years = df.columns  # Use the DataFrame columns as the years

    for year in years:
        filtered_df = df[year].dropna()  # Drop NaN values for the current year

        # Set color and title settings based on input
        if title == 'Net Import Ratio Map with slider':
            colorscale = [
                [0.0, '#1B4F72'],
                [0.3, '#2E86C1'],
                [0.45, '#AED6F1'],
                [0.55, '#D7BDE2'],
                [0.7, '#884EA0'],
                [1.0, '#512E5F']
            ]
            colorbar = dict(
                tickvals=[-1.0, -0.05, 0.05, 1.0],
                ticktext=['-100%', '-5%', '5%', '100%'],
                len=0.8,
                lenmode='fraction'
            )
            hovertemplate = '<b>Country:</b> %{location}<br>' + '<b>Net Import Ratio:</b> %{z:.2%}<extra></extra>'
            zmin, zmax = -1.0, 1.0

        elif title == 'GDP per Capita with slider':
            colorscale = [
                [0.0, '#fee5d9'],
                [1136 / 80000, '#fcae91'],  # Place 1136 at approximately 1/3rd position
                [13845 / 80000, '#de2d26'],  # Place 13845 at approximately 2/3rd position
                [1.0, '#67000d']
            ]
            colorbar = dict(
                tickvals=[1136, 13845, 80000],
                ticktext=['1136', '13845', '80000'],
                len=0.8,
                lenmode='fraction'
            )
            hovertemplate = '<b>Country:</b> %{location}<br>' + '<b>GDP per Capita:</b> %{z:.2f}<extra></extra>'
            zmin, zmax = 0, 80000

        trace = go.Choropleth(
            locations=filtered_df.index,
            z=filtered_df.values,
            locationmode='country names',
            colorscale=colorscale,
            colorbar=colorbar,
            hovertemplate=hovertemplate,
            zmin=zmin,
            zmax=zmax,
            visible=False
        )

        # Add the trace to the figure
        fig.add_trace(trace)

    # Set the first trace to visible
    if len(fig.data) > 0:
        fig.data[0].visible = True

    # Create animation steps
    steps = []
    for i, year in enumerate(years):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title_text': f'{title} - {year}', 'frame': {'duration': 1000, 'redraw': True}}],
            label=str(year)
        )
        step['args'][0]['visible'][i] = True
        steps.append(step)

    # Create the slider
    sliders = [dict(
        active=0,
        steps=steps,
        currentvalue={"prefix": "Year: ", "font": {"size": 14}},
    )]

    # Update the layout of the figure with increased size and change the template
    fig.update_layout(
        title_text=title,
        title_font_size=24,
        title_x=0.5, title_xanchor='center',
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type='natural earth',
            showcountries=True
        ),
        sliders=sliders,
        height=500,
        width=500,
        font=dict(family='Arial', size=12),
        margin=dict(t=80, l=50, r=50, b=50),
        template='plotly'
    )

    return fig

# Plot maps
try:
    fig1 = plot_world_map(net_import_ratio, 'Net Import Ratio Map with slider')
    fig2 = plot_world_map(data_imputation.xs('Economics: GDP Per Capita', level=1), 'GDP per Capita with slider')
    
    # Display the figures side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"An error occurred while plotting the maps: {e}")