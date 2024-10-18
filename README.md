# CMSE830 Midterm Project

## Exploring the Correlation between Electricity and Economy

![Description of the image](img/overview.png)

### Project Overview

The main goal of this project is to explore the correlation between global economic indicators and electricity data from 1992 to 2021 and to identify representative features that capture these relationships. Additionally, the project seeks to examine differences in electricity strategies among countries at various stages of economic development.

Key hypotheses include:

- Is GDP related to electricity generation or consumption?
- Are import and export data between neighboring countries negatively correlated, indicating regional complementarity?
- Do the three energy strategies (self-sufficient, importer, exporter) correspond to distinct economic development stages? Are high-income countries more common among exporters?

This analysis aims to provide insights into how economic and energy dynamics shape national and regional strategies.

### Datasets

This project utilizes the following datasets:

**1. World GDP(GDP, GDP per capita, and annual growths):**

https://www.kaggle.com/datasets/zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths

GDP data is gathered from World Bank. More than +200 country GDP is used between the years 1960-2020.

**2.	Global Electricity Statistics (1980-2021):**

https://www.kaggle.com/datasets/akhiljethwa/global-electricity-statistics

This dataset contains detailed statistics on global electricity production, with a focus on the different sources of electricity, including fossil fuels and renewables. It also provides insights into energy generation capacity and total electricity consumption by country.

The two datasets complement each other by providing a full picture of how energy consumption patterns have changed globally. The electricity statistics help understand the production and supply side, while the sustainable energy dataset offers a view into the consumption and environmental impact.

### Key Features

- Included IDA and EDA, with data restructuring and format adjustments.
- Conducted data cleaning and applied linear imputation, along with MAR and MCAR analysis.
- Classified countries as importers, exporters, or self-sufficient, with corresponding economic analysis.
- Grouped countries into low, middle, and high-income categories and analyzed electricity strategies in each group.
- Examined complementary electricity characteristics between neighboring countries.

### Usage

This project is implemented using [Streamlit](https://streamlit.io/), allowing users to interact with the simulation data and explore various scenarios. The app provides real-time visualizations and analysis tools to help users identify the most efficient computational strategies for their specific needs.

### Conclusion
Most countries adopt a self-sufficient electricity strategy, particularly among the very large and very small economies. Exporting countries within low- and middle-income brackets tend to have relatively higher per capita electricity infrastructure and generation capacity, along with higher distribution losses. This distinction is less evident among high-income nations.

Self-sufficient countries show a more balanced economic distribution. In these countries, high-income nations make up a larger share, though their income median is lower, while low-income nations form a smaller share but have a higher income median.

Regional electricity complementarity often appears among lower-income countries, with one country relying on its neighbor’s power supply. However, the Nordic countries are an exception, showing flexibility and frequency in switching between importer and exporter roles.

### AI Usage
Almost all the code was completed by ChatGPT-4o Canvas. My job is to explain the logic and functionality of the code in human language and check for errors in the code for correction.

Even the parts I wrote myself heavily relied on Copilot in VSCode for code completion. 

Without AI assistance, I could still complete the project, but it would be impossible to write so much code in such a short amount of time.


