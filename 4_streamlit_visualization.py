import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# import pycountry

# Load data
try:
    data_imputation = pd.read_csv('data_imputation.csv', index_col=[0, 1])
    data_cleaned = pd.read_csv('data_cleaned.csv', index_col=[0, 1])
except FileNotFoundError:
    st.error("The csv file could not be found. Please make sure it is available in the same directory.")
    data_imputation = pd.DataFrame()
    data_cleaned = pd.DataFrame()

# 1. Set up Streamlit app with multiple pages
st.set_page_config(page_title="Electricity and Economy Analysis", page_icon="📊", layout="wide")

# Sidebar for page navigation
page = st.sidebar.selectbox("Choose a page", ["Overview", "Data Analysis", "Imputation", "Conclusion"])

# Page 1: Project Overview
if page == "Overview":
    st.title("Overview")
    st.write("## Exploring the Correlation between Electricity and Economy")
    st.write("The main goal:")
    st.write("1) Explore the correlation between global economic data and electricity data to identify the representative features.")
    st.write("2) Examine differences in electricity strategies across countries at different economic stages.")

    st.write("### Let's get Familiar with the Data!")
    st.write("### Global Electrical Data")
    # Sidebar selection for country if data is not empty
    selected_country = None
    if not data_imputation.empty:
        selected_country = st.sidebar.selectbox("Select a country", sorted(data_imputation.index.get_level_values(0).unique()))

    # Filter data based on selected country
    country_data = data_imputation.loc[selected_country]
    years = [str(year) for year in range(1992, 2022)]

    # Extract data for each feature
    imports = country_data.loc['Electricity: Imports'][years].values
    exports = country_data.loc['Electricity: Exports'][years].values
    distribution_losses = country_data.loc['Electricity: Distribution Losses'][years].values
    net_consumption = country_data.loc['Electricity: Net Consumption'][years].values
    net_generation = country_data.loc['Electricity: Net Generation'][years].values

    fig = go.Figure()

    # Add a stacked area chart of net consumption
    fig.add_trace(go.Scatter(
        x=years, y=net_consumption, mode='lines', fill='tonexty', name='Net Consumption', line=dict(width=0.5, color='#F39C12'),
        customdata=net_consumption, hovertemplate='%{customdata:.2f} Million MWh'
    ))

    # Add a stacked area chart of exports
    fig.add_trace(go.Scatter(
        x=years, y=net_consumption + exports, mode='lines', fill='tonexty', name='Exports', line=dict(width=0.5, color='red'),
        customdata=exports, hovertemplate='%{customdata:.2f} Million MWh'
    ))

    # Add a stacked area chart of distribution losses
    fig.add_trace(go.Scatter(
        x=years, y=net_consumption + exports + distribution_losses, mode='lines', fill='tonexty', name='Distribution Losses', line=dict(width=0.5, color='#28B463'),
        customdata=distribution_losses, hovertemplate='%{customdata:.2f} Million MWh'
    ))

    # Add a stacked area chart of imports
    fig.add_trace(go.Scatter(
        x=years, y=net_consumption + exports + distribution_losses - imports, mode='lines', fill='tonexty', name='Imports', line=dict(width=0.5, color='blue'),
        customdata=imports, hovertemplate='%{customdata:.2f} Million MWh'
    ))

    # Add a line plot of net generation
    fig.add_trace(go.Scatter(
        x=years, y=net_generation, mode='lines+markers', name='Net Generation', 
        line=dict(width=2, color='#7D3C98'),
        hovertemplate='%{y:.2f} Million MWh'
    ))

    fig.update_layout(
        title=f'Stacked Area Chart: Net Consumption, Exports, Distribution Losses, Imports ({selected_country}, 1992-2021)',
        xaxis_title='Year',
        yaxis_title='Million MWh',
        hovermode="x unified",
        template='plotly_white',
        height=600,
    )

    # Update y-axis to automatically set the range
    fig.update_yaxes(automargin=True, range=[net_consumption.min() * 0.9, (net_consumption + exports + distribution_losses).max() * 1.1])

    st.plotly_chart(fig)

    # Add Economic Data Plots
    st.write("### Global Economic Data")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Extract economic data
    gdp = country_data.loc['Economics: GDP'][years].values
    ppp = country_data.loc['Economics: PPP'][years].values
    gdp_growth = country_data.loc['Economics: GDP Growth'][years].values
    ppp_growth = country_data.loc['Economics: PPP Growth'][years].values
    gdp_per_capita = country_data.loc['Economics: GDP Per Capita'][years].values
    ppp_per_capita = country_data.loc['Economics: PPP Per Capita'][years].values

    # Create Economic Data Figure using Plotly
    economic_fig = go.Figure()

    # Add bar plot for GDP
    economic_fig.add_trace(go.Bar(
        x=years,
        y=gdp,
        customdata=gdp/1e9,
        name='GDP',
        marker=dict(color='#5DADE2'),
        hovertemplate='%{customdata:.1f} Billion USD'
    ))

    # Add bar plot for PPP
    economic_fig.add_trace(go.Bar(
        x=years,
        y=ppp,
        customdata=ppp/1e9,
        name='PPP',
        marker=dict(color='#2E86C1'),
        hovertemplate='%{customdata:.1f} Billion USD'
    ))

    # Add line plot for GDP Growth
    economic_fig.add_trace(go.Scatter(
        x=years,
        y=gdp_growth,
        mode='lines+markers',
        name='GDP Growth',
        yaxis='y2',
        marker=dict(size=8, color='#EB984E', symbol='circle'),
        line=dict(width=2, color='#EB984E'),
        hovertemplate='%{y:.1f}%'
    ))

    # Add line plot for PPP Growth
    economic_fig.add_trace(go.Scatter(
        x=years,
        y=ppp_growth,
        mode='lines+markers',
        name='PPP Growth',
        yaxis='y2',
        marker=dict(size=8, color='#E67E22', symbol='x'),
        line=dict(width=2, color='#E67E22'),
        hovertemplate='%{y:.1f}%'
    ))

    economic_fig.update_layout(
        title=f'GDP, PPP, GDP Growth, and PPP Growth ({selected_country}, 1992-2021)',
        xaxis_title='Year',
        yaxis=dict(title='Billion USD', showgrid=True, zeroline=False),
        yaxis2=dict(title='Growth Rate (%)', overlaying='y', side='right', showgrid=True, gridcolor='grey', zeroline=False),
        hovermode="x unified",
        barmode='group',  # Group the bar plots side by side
        template='plotly_white',
        xaxis=dict(tickangle=-45),
        height=500
    )

    st.plotly_chart(economic_fig)

    # Right subplot: GDP Per Capita, PPP Per Capita
    per_capita_fig = go.Figure()

    # Add bar plot for GDP Per Capita
    per_capita_fig.add_trace(go.Bar(
        x=years,
        y=gdp_per_capita,
        name='GDP Per Capita',
        marker=dict(color='#48C9B0'),
        hovertemplate='%{y:.1f} USD'
    ))

    # Add bar plot for PPP Per Capita
    per_capita_fig.add_trace(go.Bar(
        x=years,
        y=ppp_per_capita,
        name='PPP Per Capita',
        marker=dict(color='#148F77'),
        hovertemplate='%{y:.1f} USD'
    ))

    per_capita_fig.update_layout(
        title=f'GDP Per Capita and PPP Per Capita ({selected_country}, 1992-2021)',
        xaxis_title='Year',
        yaxis_title='USD per Capita',
        hovermode="x unified",
        barmode='group',  # Group the bar plots side by side
        template='plotly_white',
        xaxis=dict(tickangle=-45),
        height=500
    )

    st.plotly_chart(per_capita_fig)

    # Display the features table
    st.write("## Features of the Datasets")
    st.write("The original idea was that clicking options in the table would trigger corresponding changes in the data displayed on the map below. This part will be implemented in the final project.")
    st.write(":(")

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
                    [1000 / 80000, '#fcae91'],  # Place 1000 at approximately 1/3rd position
                    [12000 / 80000, '#de2d26'],  # Place 12000 at approximately 2/3rd position
                    [1.0, '#67000d']
                ]
                colorbar = dict(
                    tickvals=[1000, 12000, 80000],
                    ticktext=['1000', '12000', '80000'],
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

    st.write("## Data Structure")
    st.write("The data is organized using a multi-index structure, resembling a four-dimensional matrix where each entry corresponds to a specific value based on country, feature, and year.")
    st.image("img/data_structure.png")

    st.write("## Some Hypothesis")
    st.image("img/hypothesis.png")

# Page 2: Correlation Analysis
elif page == "Data Analysis":
    # Page: Correlation Analysis
    st.title("Data Analysis")
    st.write("## Heatmap")
    st.write('"Total value" refers to the result obtained by adding up data from all countries.')
    st.write('"Normalized value" refers to the result obtained after normalizing the data from each country.')

    # 2. Heatmap Analysis
    # 2.1 Calculate the total values for each feature across all countries
    # Sum the values for each feature independently
    total_values = data_imputation.groupby(level=1).sum()

    # Calculate the correlation matrix for total values
    correlation_total = total_values.T.corr()

    # 2.2 Normalize data for each country to ensure equal weight for all countries
    normalized_data = data_imputation.groupby(level=0).transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Calculate the average values for each feature across all countries after normalization
    average_normalized_values = normalized_data.groupby(level=1).mean()

    # Calculate the correlation matrix for average normalized values
    correlation_average = average_normalized_values.T.corr()

    # 2.3 Plot the heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Extract different types of features
    total_features = [col for col in data_imputation.index.levels[1] if not ('Per Capita' in col or 'Growth' in col)]
    per_capita_features = [col for col in data_imputation.index.levels[1] if 'Per Capita' in col]
    growth_features = [col for col in data_imputation.index.levels[1] if 'Growth' in col]

    # Calculate correlation matrices for total, per capita, and growth features
    try:
        total_corr = data_imputation.loc[(slice(None), total_features), :].groupby(level=1).sum().T.corr()
        per_capita_corr = data_imputation.loc[(slice(None), per_capita_features), :].groupby(level=1).sum().T.corr()

        gdp_values = data_imputation.xs('Economics: GDP', level=1)
        weights = gdp_values.div(gdp_values.sum(axis=0), axis=1)
        growth_weighted_sum = data_imputation.loc[(slice(None), growth_features), :].mul(weights, level=0).groupby(level=1).sum()
        growth_corr = growth_weighted_sum.T.corr()
    except KeyError as e:
        st.error(f"An error occurred while calculating correlations: {e}")
        total_corr = per_capita_corr = growth_corr = pd.DataFrame()

    # Calculate correlation matrices for normalized average values
    total_avg_corr = normalized_data.loc[(slice(None), total_features), :].groupby(level=1).mean().T.corr()
    per_capita_avg_corr = normalized_data.loc[(slice(None), per_capita_features), :].groupby(level=1).mean().T.corr()
    growth_avg_corr = normalized_data.loc[(slice(None), growth_features), :].groupby(level=1).mean().T.corr()

    # Custom label function to remove prefixes
    def custom_label(label):
        return label.replace('Electricity: ', '').replace('Economics: ', '')

    # Plot specific parts of heatmaps
    x_subset1 = ['Economics: GDP', 'Economics: PPP']
    y_subset1 = ['Electricity: Distribution Losses', 'Electricity: Exports', 'Electricity: Imports', 'Electricity: Installed Capacity', 'Electricity: Net Consumption', 'Electricity: Net Generation', 'Electricity: Net Imports']
    if not total_corr.empty:
        sns.heatmap(total_corr.loc[y_subset1, x_subset1].rename(index=custom_label, columns=custom_label), annot=True, cmap='YlGnBu', ax=axes[0, 0], vmin=-1, vmax=1, annot_kws={"size": 14})
    axes[0, 0].set_title('Correlation Heatmap of Total Feature Values')
    axes[0, 0].text(-0.5, 1.02, f'{1}', transform=axes[0, 0].transAxes, fontsize=18, verticalalignment='top', fontweight='bold')
    axes[0, 0].set_aspect(aspect=0.5)

    x_subset2 = ['Economics: GDP Per Capita', 'Economics: PPP Per Capita']
    y_subset2 = ['Electricity: Distribution Losses Per Capita', 'Electricity: Exports Per Capita', 'Electricity: Imports Per Capita', 'Electricity: Installed Capacity Per Capita', 'Electricity: Net Consumption Per Capita', 'Electricity: Net Generation Per Capita', 'Electricity: Net Imports Per Capita']
    if not per_capita_corr.empty:
        sns.heatmap(per_capita_corr.loc[y_subset2, x_subset2].rename(index=custom_label, columns=custom_label), annot=True, cmap='YlGnBu', ax=axes[0, 1], vmin=-1, vmax=1, annot_kws={"size": 14})
    axes[0, 1].set_title('Correlation Heatmap of Per Capita Feature Values')
    axes[0, 1].text(-0.5, 1.02, f'{2}', transform=axes[0, 1].transAxes, fontsize=18, verticalalignment='top', fontweight='bold')
    axes[0, 1].set_aspect(aspect=0.5)

    x_subset3 = ['Economics: GDP Growth', 'Economics: PPP Growth']
    y_subset3 = ['Electricity: Distribution Losses Growth', 'Electricity: Exports Growth', 'Electricity: Imports Growth', 'Electricity: Installed Capacity Growth', 'Electricity: Net Consumption Growth', 'Electricity: Net Generation Growth', 'Electricity: Net Imports Growth']
    if not growth_corr.empty:
        sns.heatmap(growth_corr.loc[y_subset3, x_subset3].rename(index=custom_label, columns=custom_label), annot=True, cmap='YlGnBu', ax=axes[0, 2], vmin=-1, vmax=1, annot_kws={"size": 14})
    axes[0, 2].set_title('Correlation Heatmap of Growth Feature Values')
    axes[0, 2].text(-0.5, 1.02, f'{3}', transform=axes[0, 2].transAxes, fontsize=18, verticalalignment='top', fontweight='bold')
    axes[0, 2].set_aspect(aspect=0.5)

    sns.heatmap(total_avg_corr.loc[y_subset1, x_subset1].rename(index=custom_label, columns=custom_label), annot=True, cmap='YlGnBu', ax=axes[1, 0], vmin=-1, vmax=1, annot_kws={"size": 14})
    axes[1, 0].set_title('Correlation Heatmap of Average Normalized Total Values')
    axes[1, 0].text(-0.5, 1.02, f'{4}', transform=axes[1, 0].transAxes, fontsize=18, verticalalignment='top', fontweight='bold')
    axes[1, 0].set_aspect(aspect=0.5)

    sns.heatmap(per_capita_avg_corr.loc[y_subset2, x_subset2].rename(index=custom_label, columns=custom_label), annot=True, cmap='YlGnBu', ax=axes[1, 1], vmin=-1, vmax=1, annot_kws={"size": 14})
    axes[1, 1].set_title('Correlation Heatmap of Average Normalized Per Capita Values')
    axes[1, 1].text(-0.5, 1.02, f'{5}', transform=axes[1, 1].transAxes, fontsize=18, verticalalignment='top', fontweight='bold')
    axes[1, 1].set_aspect(aspect=0.5)

    sns.heatmap(growth_avg_corr.loc[y_subset3, x_subset3].rename(index=custom_label, columns=custom_label), annot=True, cmap='YlGnBu', ax=axes[1, 2], vmin=-1, vmax=1, annot_kws={"size": 14})
    axes[1, 2].set_title('Correlation Heatmap of Average Normalized Growth Values')
    axes[1, 2].text(-0.5, 1.02, f'{6}', transform=axes[1, 2].transAxes, fontsize=18, verticalalignment='top', fontweight='bold')
    axes[1, 2].set_aspect(aspect=0.5)

    # Display the heatmaps in Streamlit
    st.pyplot(fig)

    st.write("Subplots 1 and 2 confirm strong economic-electricity correlations. Normalization in subplots 4 and 5 weakens these, with GDP even turning negative.")
    st.write("Direct electricity data (subplot 3) shows strong economic ties, while indirect factors (imports, exports) are weaker, likely due to other influences.")
    st.write("The growth rate of GDP and PPP per capita show a strong positive correlation with electricity data.")

    # 4. Scatter Plot Analysis
    st.write("## Scatter Plot & Histogram")
    net_imports = data_imputation.xs('Electricity: Net Imports', level=1)
    net_generation = data_imputation.xs('Electricity: Net Generation', level=1).replace(0, np.nan)  # Avoid division by zero
    net_import_ratio = net_imports.div(net_generation)

    # 4.1 Classify energy status based on yearly net import ratio
    net_import_ratio_threshold = 0.05
    classification_threshold = 0.7

    def classify_energy_status(country_data):
        years_as_exporter = (country_data < -net_import_ratio_threshold).sum()
        years_as_importer = (country_data > net_import_ratio_threshold).sum()
        years_as_self_sufficient = ((country_data >= -0.05) & (country_data <= 0.05)).sum()
        total_years = len(country_data)
        
        if total_years == 0:
            return 'Other'
        
        exporter_ratio = years_as_exporter / total_years
        importer_ratio = years_as_importer / total_years
        self_sufficient_ratio = years_as_self_sufficient / total_years
        switcher_ratio = (years_as_importer + years_as_exporter) / total_years
        
        if exporter_ratio > classification_threshold:
            return 'Net Energy Exporter'
        elif importer_ratio > classification_threshold:
            return 'Net Energy Importer'
        elif self_sufficient_ratio > classification_threshold:
            return 'Energy Self-Sufficient'
        elif switcher_ratio > classification_threshold:
            return 'Frequent Import-Export Switcher'
        else:
            return 'Other'

    # Apply classification to each country
    energy_status = net_import_ratio.groupby(level=0).apply(lambda group: classify_energy_status(group.values))

    # Split data into groups based on classification
    importer_countries = energy_status[energy_status == 'Net Energy Importer'].index
    exporter_countries = energy_status[energy_status == 'Net Energy Exporter'].index
    self_sufficient_countries = energy_status[energy_status == 'Energy Self-Sufficient'].index
    switcher_countries = energy_status[energy_status == 'Frequent Import-Export Switcher'].index
    other_countries = energy_status[energy_status == 'Other'].index

    data_importers = data_imputation.loc[importer_countries]
    data_exporters = data_imputation.loc[exporter_countries]
    data_self_sufficient = data_imputation.loc[self_sufficient_countries]
    data_switchers = data_imputation.loc[switcher_countries]
    data_others = data_imputation.loc[other_countries]

    # 4.2 Plot
    # Define y-axis limits for each feature
    y_limits = {
        'Electricity: Distribution Losses Per Capita': (1e-10, 1e-5),
        'Electricity: Installed Capacity Per Capita': (1e-9, 1e-5),
        'Electricity: Net Consumption Per Capita': (1e-9, 1e-4),
        'Electricity: Net Generation Per Capita': (1e-9, 1e-4),
    }

    y_limits_2 = {
        'Electricity: Distribution Losses': (1e-4, 1e3),
        'Electricity: Installed Capacity': (1e-3, 1e4),
        'Electricity: Net Consumption': (1e-3, 1e4),
        'Electricity: Net Generation': (1e-3, 1e4),
    }

    electricity_features = ['Electricity: Distribution Losses Per Capita',  
                            'Electricity: Installed Capacity Per Capita', 
                            'Electricity: Net Consumption Per Capita', 
                            'Electricity: Net Generation Per Capita']

    electricity_features_2 = ['Electricity: Distribution Losses',  
                            'Electricity: Installed Capacity', 
                            'Electricity: Net Consumption', 
                            'Electricity: Net Generation']

    def plot_scatter(ax, x_feature, y_feature, title, y_lim=None):
        if not data_self_sufficient.empty:
            ax.scatter(data_self_sufficient.xs(x_feature, level=1), 
                    data_self_sufficient.xs(y_feature, level=1), 
                    color='grey', label='Self-Sufficient', alpha=0.2)
        if not data_importers.empty:
            ax.scatter(data_importers.xs(x_feature, level=1), 
                    data_importers.xs(y_feature, level=1), 
                    color='sandybrown', label='Net Importer', alpha=0.6)
        if not data_exporters.empty:
            ax.scatter(data_exporters.xs(x_feature, level=1), 
                    data_exporters.xs(y_feature, level=1), 
                    color='forestgreen', label='Net Exporter', alpha=0.8)
        if not data_switchers.empty:
            ax.scatter(data_switchers.xs(x_feature, level=1), 
                    data_switchers.xs(y_feature, level=1), 
                    color='red', label='Switcher', alpha=0.8)
        if not data_others.empty:
            ax.scatter(data_others.xs(x_feature, level=1), 
                    data_others.xs(y_feature, level=1), 
                    color='blue', label='Others', alpha=0.8)
        
        ax.set_xlabel(x_feature.replace('Economics: ', '') + ' (in USD)')
        ax.set_ylabel(y_feature.replace('Electricity: ', '') + ' (in million MWh)')
        ax.set_xscale('log')  # Set x-axis to log scale
        ax.set_yscale('log')  # Set y-axis to log scale
        if y_lim:
            ax.set_ylim(y_lim)  # Set y-axis limits
        ax.set_title(title)
        if ax.get_legend_handles_labels()[1]:  # Only add legend if there are labels
            ax.legend()
        ax.grid(True)

    def add_regression_line(ax, x_data, y_data, color, label):
        # Flatten the data and remove zero values
        x_data = x_data.values.flatten()
        y_data = y_data.values.flatten()
        valid_mask = (x_data > 0) & (y_data > 0)
        x_data, y_data = x_data[valid_mask], y_data[valid_mask]

        # Check if there are enough data points for regression
        if len(x_data) > 1 and len(y_data) > 1:
            # Fit a linear regression model in log-log space
            coeffs = np.polyfit(np.log10(x_data), np.log10(y_data), 1)
            fit_y = 10 ** (coeffs[0] * np.log10(x_data) + coeffs[1])  # Calculate the fitted y values
            ax.plot(x_data, fit_y, color=color, linestyle='--', linewidth=3, label=f'{label} Trend')

    def add_transparent_overlay(ax, x_range, y_range, alpha=0.25):
        ax.fill_between(x_range, y_range[0], y_range[1], color='lightsteelblue', alpha=alpha)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    income_thresholds = {
        1992: {'low_income': 675, 'high_income': 8325},
        1993: {'low_income': 695, 'high_income': 8645},
        1994: {'low_income': 725, 'high_income': 8955},
        1995: {'low_income': 765, 'high_income': 9355},
        1996: {'low_income': 785, 'high_income': 9635},
        1997: {'low_income': 785, 'high_income': 9655},
        1998: {'low_income': 760, 'high_income': 9855},
        1999: {'low_income': 755, 'high_income': 9955},
        2000: {'low_income': 755, 'high_income': 9955},
        2001: {'low_income': 745, 'high_income': 9355},
        2002: {'low_income': 735, 'high_income': 9355},
        2003: {'low_income': 735, 'high_income': 9655},
        2004: {'low_income': 765, 'high_income': 10065},
        2005: {'low_income': 875, 'high_income': 10365},
        2006: {'low_income': 905, 'high_income': 10655},
        2007: {'low_income': 935, 'high_income': 10855},
        2008: {'low_income': 975, 'high_income': 11355},
        2009: {'low_income': 995, 'high_income': 11655},
        2010: {'low_income': 1005, 'high_income': 12275},
        2011: {'low_income': 1025, 'high_income': 12375},
        2012: {'low_income': 1035, 'high_income': 12575},
        2013: {'low_income': 1045, 'high_income': 12615},
        2014: {'low_income': 1045, 'high_income': 12735},
        2015: {'low_income': 1045, 'high_income': 12635},
        2016: {'low_income': 1005, 'high_income': 12475},
        2017: {'low_income': 995, 'high_income': 12475},
        2018: {'low_income': 995, 'high_income': 12375},
        2019: {'low_income': 1025, 'high_income': 12375},
        2020: {'low_income': 1035, 'high_income': 12535},
        2021: {'low_income': 1045, 'high_income': 12695}
    }

    # Plot scatter maps for GDP Per Capita and PPP Per Capita
    for i, feature in enumerate(electricity_features):
        plot_scatter(axes[i], 'Economics: GDP Per Capita', feature, f'GDP Per Capita vs {feature.replace("Electricity: ", "")}', y_lim=y_limits.get(feature))
        # Add transparent overlay to subplots 1-4
        # add_transparent_overlay(axes[i], x_range=(1136, 13845), y_range=y_limits.get(feature)) # middle income countries
        # Add subplot label with numeric index
        axes[i].text(-0.12, 1, f'{i + 1}', transform=axes[i].transAxes, fontsize=14, verticalalignment='top', fontweight='bold')
        # Add regression lines
        if not data_self_sufficient.empty:
            add_regression_line(axes[i], data_self_sufficient.xs('Economics: GDP Per Capita', level=1), 
                                data_self_sufficient.xs(feature, level=1), '#717D7E', 'Self-Sufficient')
        if not data_importers.empty:
            add_regression_line(axes[i], data_importers.xs('Economics: GDP Per Capita', level=1), 
                                data_importers.xs(feature, level=1), '#B7950B', 'Net Importer')
        if not data_exporters.empty:
            add_regression_line(axes[i], data_exporters.xs('Economics: GDP Per Capita', level=1), 
                                data_exporters.xs(feature, level=1), '#196F3D', 'Net Exporter')

    for i, feature in enumerate(electricity_features_2):
        plot_scatter(axes[i + 4], 'Economics: GDP', feature, f'GDP vs {feature.replace("Electricity: ", "")}', y_lim=y_limits_2.get(feature))
        # Add subplot label with numeric index
        axes[i + 4].text(-0.12, 1, f'{i + 5}', transform=axes[i + 4].transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

    # Hide any unused subplots
    for j in range(len(electricity_features)+len(electricity_features_2), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    st.pyplot(fig)

    # print(f"Net Importer Countries: {len(importer_countries)}")
    # print(f"Net Exporter Countries: {len(exporter_countries)}")
    # print(f"Self-Sufficient Countries: {len(self_sufficient_countries)}")
    # print(f"Switcher Countries: {len(switcher_countries)}")
    # print(f"Other Countries: {len(other_countries)}")

    # 4.2 Calculate GDP per capita distribution for each group
    gdp_per_capita = data_imputation.xs('Economics: GDP Per Capita', level=1)

    def calculate_gdp_distribution(group_data, year):
        thresholds = income_thresholds[year]
        low_income_threshold = thresholds['low_income']
        high_income_threshold = thresholds['high_income']
        
        med_group_data = group_data.median(axis=1)  # Calculate average GDP per capita for each country over the years
        below_low_income = (med_group_data < low_income_threshold).sum() / len(med_group_data)
        between_low_high_income = ((med_group_data >= low_income_threshold) & (med_group_data <= high_income_threshold)).sum() / len(med_group_data)
        above_high_income = (med_group_data > high_income_threshold).sum() / len(med_group_data)
        
        # Normalize median values using Min-Max normalization within each group
        def min_max_normalize(data):
            min_value = data.min() if data.min() != data.max() else 0
            max_value = data.max() if data.max() != 0 else 1
            return (data.median() - min_value) / (max_value - min_value) if not data.empty else 0
        
        med_below_low_income = min_max_normalize(med_group_data[med_group_data < low_income_threshold])
        med_between_low_high_income = min_max_normalize(med_group_data[(med_group_data >= low_income_threshold) & (med_group_data <= high_income_threshold)])
        med_above_high_income = min_max_normalize(med_group_data[med_group_data > high_income_threshold])
        
        return float(below_low_income), float(between_low_high_income), float(above_high_income), med_below_low_income, med_between_low_high_income, med_above_high_income

    # Calculate distribution for each group for a specific year
    year = 2021  # can change the year as needed
    gdp_importers = list(calculate_gdp_distribution(gdp_per_capita.loc[importer_countries], year))
    gdp_exporters = list(calculate_gdp_distribution(gdp_per_capita.loc[exporter_countries], year))
    gdp_self_sufficient = list(calculate_gdp_distribution(gdp_per_capita.loc[self_sufficient_countries], year))

    categories = ['Low Income', 'Middle Income', 'High Income']
    x = np.arange(len(categories))
    width = 0.1

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot GDP per capita distribution
    ax.bar(x - width - 0.2, gdp_importers[:3], width, label='Net Electricity Importers', color='sandybrown')
    ax.bar(x - 0.05, gdp_exporters[:3], width, label='Net Electricity Exporters', color='forestgreen')
    ax.bar(x + width + 0.1, gdp_self_sufficient[:3], width, label='Electricity Self-Sufficient', color='grey')

    # Plot average GDP per capita for each income category
    avg_gdp_values_importers = gdp_importers[3:]
    avg_gdp_values_exporters = gdp_exporters[3:]
    avg_gdp_values_self_sufficient = gdp_self_sufficient[3:]

    ax2 = ax.twinx()  # Create a secondary y-axis
    bar1 = ax2.bar(x - width - 0.1, avg_gdp_values_importers, width, color='#FAD7A0', alpha=0.7, label='Median (Importers)')
    bar2 = ax2.bar(x + 0.05, avg_gdp_values_exporters, width, color='#A9DFBF', alpha=0.7, label='Median (Exporters)')
    bar3 = ax2.bar(x + width + 0.2, avg_gdp_values_self_sufficient, width, color='#CCD1D1', alpha=0.7, label='Median (Self-Sufficient)')
    ax2.set_ylabel('Normalized Median GDP per Capita (USD)')

    # Set labels and title
    ax.set_xlabel('Income Category')
    ax.set_ylabel('Share of Total Electricity (%)')
    ax.set_title('GDP per Capita Distribution by Energy Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.set_ylim(0,0.6)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0,0.8)

    fig.tight_layout()
    st.pyplot(fig)

    st.write("The distribution of importers and self-sufficient countries is similar, while exporters tend to have more electricity infrastructure and generation capacity. In some low- and middle-income countries, exporters show notably higher electricity capacity than other nations at the same economic level, though this difference is less evident in high-income countries.")

    st.write("Both importers and exporters are mainly concentrated in mid- and small-sized economies.")

    st.write("Closer examination reveals that self-sufficient, low-income countries have a higher median per capita GDP, while high-income self-sufficient countries have a lower median, suggesting smaller income disparities. Self-sufficient countries also include more high-income nations and fewer low-income ones.")

    st.write("Exporters show the opposite trend, hinting at a possible link between electricity strategies and economic health.")

# Page 3: Imputation
elif page == "Imputation":
    st.title("Data Cleaning and Imputation")
    st.write("## Linear Imputation")

    # Step 1: Remove countries with over 9 years missing data for any feature
    st.write("### Step 1: Removing Countries with Excessive Missing Data")
    st.write("I removed countries that had a single feature missing for more than 9 years during the period from 1992 to 2021. This is because more than 1/3 missing data is considered too much to impute accurately.")

    # Step 2: MAR and MCAR testing
    st.write("### Step 2: MAR and MCAR Testing")
    st.write("I conducted tests to determine whether the missing data mechanism was MCAR or MAR. The results indicated that the missing data is highly correlated with other variables, meaning it is MAR.")

    # Step 3: Imputation Techniques
    st.write("### Step 3: Imputation Techniques")
    st.write("I used a combination of linear interpolation and forward/backward filling to impute missing data. Given that the main trend in the data is either increasing or stable over time, using neighboring years to fill in missing values was deemed appropriate.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2: st.image("img/linear_imputation.png", width=700)
    st.caption("References from: doi.org/10.1002/we.2888")

    st.write("## Missing Data Distribution")
    # 3.1. Group countries based on GDP per capita
    gdp_per_capita = data_cleaned.xs('Economics: GDP Per Capita', level=1)
    median_gdp_per_country = gdp_per_capita.median(axis=1)

    # Define GDP thresholdd
    income_thresholds = {
        1992: {'low_income': 675, 'high_income': 8325},
        1993: {'low_income': 695, 'high_income': 8645},
        1994: {'low_income': 725, 'high_income': 8955},
        1995: {'low_income': 765, 'high_income': 9355},
        1996: {'low_income': 785, 'high_income': 9635},
        1997: {'low_income': 785, 'high_income': 9655},
        1998: {'low_income': 760, 'high_income': 9855},
        1999: {'low_income': 755, 'high_income': 9955},
        2000: {'low_income': 755, 'high_income': 9955},
        2001: {'low_income': 745, 'high_income': 9355},
        2002: {'low_income': 735, 'high_income': 9355},
        2003: {'low_income': 735, 'high_income': 9655},
        2004: {'low_income': 765, 'high_income': 10065},
        2005: {'low_income': 875, 'high_income': 10365},
        2006: {'low_income': 905, 'high_income': 10655},
        2007: {'low_income': 935, 'high_income': 10855},
        2008: {'low_income': 975, 'high_income': 11355},
        2009: {'low_income': 995, 'high_income': 11655},
        2010: {'low_income': 1005, 'high_income': 12275},
        2011: {'low_income': 1025, 'high_income': 12375},
        2012: {'low_income': 1035, 'high_income': 12575},
        2013: {'low_income': 1045, 'high_income': 12615},
        2014: {'low_income': 1045, 'high_income': 12735},
        2015: {'low_income': 1045, 'high_income': 12635},
        2016: {'low_income': 1005, 'high_income': 12475},
        2017: {'low_income': 995, 'high_income': 12475},
        2018: {'low_income': 995, 'high_income': 12375},
        2019: {'low_income': 1025, 'high_income': 12375},
        2020: {'low_income': 1035, 'high_income': 12535},
        2021: {'low_income': 1045, 'high_income': 12695}
    }

    year = 2021
    thresholds = income_thresholds[year]
    low_income_threshold = thresholds['low_income']
    high_income_threshold = thresholds['high_income']

    # Create groups
    low_income_countries = median_gdp_per_country[median_gdp_per_country < low_income_threshold].index
    middle_income_countries = median_gdp_per_country[(median_gdp_per_country >= low_income_threshold) & (median_gdp_per_country <= high_income_threshold)].index
    high_income_countries = median_gdp_per_country[median_gdp_per_country > high_income_threshold].index

    # Count missing values for each country across all features
    missing_counts_per_country = data_cleaned.isnull().groupby(level=0).sum().sum(axis=1)

    # Calculate missing data counts for each group and remove outliers, exclude countries with no missing values
    low_income_missing = missing_counts_per_country.loc[low_income_countries]
    low_income_missing = low_income_missing[low_income_missing > 0]
    low_income_missing = low_income_missing[low_income_missing <= low_income_missing.quantile(0.95)]

    middle_income_missing = missing_counts_per_country.loc[middle_income_countries]
    middle_income_missing = middle_income_missing[middle_income_missing > 0]
    middle_income_missing = middle_income_missing[middle_income_missing <= middle_income_missing.quantile(0.95)]

    high_income_missing = missing_counts_per_country.loc[high_income_countries]
    high_income_missing = high_income_missing[high_income_missing > 0]
    high_income_missing = high_income_missing[high_income_missing <= high_income_missing.quantile(0.95)]

    # Create a DataFrame for violin plot visualization
    grouped_missing_data = pd.DataFrame({
        'Low Income': low_income_missing,
        'Middle Income': middle_income_missing,
        'High Income': high_income_missing
    })

    # Plot violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=grouped_missing_data, orient='v')
    plt.xlabel('Income Group (2021 Standard)')
    plt.ylabel('Number of Missing Values (All 13 features)')
    plt.title('Missing Data Distribution by Income Group (Outliers Removed)')
    st.pyplot(fig)

    st.write("With data-deficient countries removed, the analysis focuses mainly on upper-middle-income countries, leading to potential underrepresentation of their characteristics.")

    st.write("In the heatmap, GDP growth rate correlation with electricity data shifted from weak to strong. Other features also show heightened correlations, indicating further amplification of existing trends.")

elif page == "Conclusion":
    st.title("Summary")
    st.write("## Electricity Strategies")
    st.write("Most countries adopt a self-sufficient electricity strategy, particularly among the very large and very small economies. Exporting countries within low- and middle-income brackets tend to have relatively higher per capita electricity infrastructure and generation capacity, along with higher distribution losses. This distinction is less evident among high-income nations.")

    st.write("Self-sufficient countries show a more balanced economic distribution. In these countries, high-income nations make up a larger share, though their income median is lower, while low-income nations form a smaller share but have a higher income median.")

    st.write("## Regional complementarity")
    st.write("Regional electricity complementarity often appears among lower-income countries, with one country relying on its neighbor's power supply. However, the Nordic countries are an exception, showing flexibility and frequency in switching between importer and exporter roles.")
