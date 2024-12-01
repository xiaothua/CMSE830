# CMSE830 Final Project (2024 Fall)

## Predicting Economic Data Using Electricity and Population Data

![Description of the image](img/overview.png)

### Project Overview

**1. Midterm Phase: Correlation Analysis**

- Explored the correlation between electricity data and economic data.

**2. Final Phase: Model Training and Prediction**

- Trained AR and VAR models separately for each country using parallel computing. Feature selection was performed for the VAR model to reduce the impact of confounding data.

**2.1. Additional Content**

- Planned to include Random Forest Regression but only partially completed it. Focused on PCA before training and compared test errors of linear regression and random forest models. 

### Datasets

This project utilizes the following datasets:

**1. World GDP (GDP, GDP per capita, and annual growths):**

https://www.kaggle.com/datasets/zgrcemta/world-gdpgdp-gdp-per-capita-and-annual-growths

GDP data is gathered from World Bank. More than +200 country GDP is used between the years 1960-2020.

**2. Global Electricity Statistics (1980-2021):**

https://www.kaggle.com/datasets/akhiljethwa/global-electricity-statistics

This dataset contains detailed statistics on global electricity production, with a focus on the different sources of electricity, including fossil fuels and renewables. It also provides insights into energy generation capacity and total electricity consumption by country.

**3.  World Population Dataset:**

https://www.kaggle.com/datasets/iamsouravbanerjee/world-population-dataset

Population data encompasses total population, growth rate, and global population share for each country, serving as potential predictive features alongside electricity data.

### Key Features

- Included IDA and EDA, with data restructuring and format adjustments.
- Conducted data cleaning and applied linear imputation, along with MAR and MCAR analysis.
- Employed feature selection methods to exclude low-relevance and confounding features.
- Applied dimensionality reduction (e.g., PCA) to create a parsimonious model.
- Automated selection of the optimal p-value (minimizing MSE) for AR and VAR models, with batch training for 170 countries using parallel computing.
- VAR model allows real-time adjustment of training set size to observe model behavior.

### Conclusion

**Midterm Phase: Correlation Analysis**

- Most countries adopt a self-sufficient electricity strategy, particularly among the very large and very small economies. Exporting countries within low- and middle-income brackets tend to have relatively higher per capita electricity infrastructure and generation capacity, along with higher distribution losses. This distinction is less evident among high-income nations.
<br/>
- Self-sufficient countries show a more balanced economic distribution. In these countries, high-income nations make up a larger share, though their income median is lower, while low-income nations form a smaller share but have a higher income median.
<br/>
- Regional electricity complementarity often appears among lower-income countries, with one country relying on its neighbor’s power supply. However, the Nordic countries are an exception, showing flexibility and frequency in switching between importer and exporter roles.

**Final Phase: Model Training and Prediction**

- Electricity and population data can partially predict GDP trends for the coming years but have significant limitations, especially during periods of economic instability (e.g., COVID-19, financial crises, wars, or trade organization exits).
<br/>
- VAR model predictions are not always better than AR models in terms of errors and MSE, though intuitively, I often prefer some VAR predictions.
<br/>
- The training set size greatly impacts results, similar to p-value selection. Each model has its optimal value, which can only be roughly estimated before try-and-error testing.

### Usage

This project is implemented using [Streamlit](https://streamlit.io/), allowing users to interact with the simulation data and explore various scenarios. The app provides real-time visualizations and analysis tools to help users identify the most efficient computational strategies for their specific needs.

### AI Usage
Almost all the code was completed by ChatGPT-4o and GPT-o1. My job is to explain the logic and functionality of the code in human language and check for errors in the code for correction.

Without AI assistance, I could still complete the project, but it would be impossible to write so much code in such a short amount of time.


