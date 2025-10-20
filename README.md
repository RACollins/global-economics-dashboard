# Global Economics Dashboard

**Live App:** [https://global-economics-dashboard.streamlit.app/](https://global-economics-dashboard.streamlit.app/)

A comprehensive interactive dashboard for exploring global economic data across multiple dimensions, from modern salary comparisons to historical economic trends spanning centuries.

## Overview

This Streamlit application provides an interactive platform for visualizing and analyzing various economic indicators across countries and time periods. The dashboard includes modern economic data (GDP, government spending, salaries, income distribution) as well as historical data stretching back to medieval England.

## Features

### Interactive Visualizations
- **Scatter plots** with customizable axes (linear or logarithmic)
- **Line charts** for time-series analysis
- **Heatmaps** for density visualization
- **Trendlines** with OLS (Ordinary Least Squares) regression
- **Population weighting** options for more representative analysis
- **Country filtering** and labeling capabilities
- **Download functionality** for all visualizations as CSV files

### Sidebar Controls
- **Log Scale Toggle**: Switch between linear and logarithmic scales for both X and Y axes
- **Population Display**: Show country populations as marker sizes in scatter plots
- **Population Weighting**: Weight trendlines by population for more representative trends
- **Country Labels**: Add labels to specific countries on plots for easier identification
- **Country Filtering**: Remove specific countries from visualizations

## Dashboard Tabs

### 1. Spending & Growth
Analyzes the relationship between government expenditure and economic growth.

**Features:**
- Time-series line plots showing GDP per capita and government expenditure over time
- Scatter plots comparing average government spending (as % of GDP) vs. annualized GDP growth
- Heatmap visualization option for density analysis
- Region averages mode to compare continental trends
- Debt adjustment mode to account for public debt in GDP calculations
- Customizable time ranges (1850-2022)
- Adjustable subperiod length for analyzing different time windows

**Data Visualized:**
- GDP per capita (Our World in Data)
- Government expenditure as % of GDP (IMF, Wikipedia, Statistica)
- Public debt as % of GDP
- Regional groupings (Asia, Americas, Africa, Europe, Oceania)

### 2. Salaries
Compares professional salaries across countries relative to GDP per capita.

**Features:**
- Job selection: Bricklayer, Doctor, Nurse, or All professions
- Scatter plot of median/mean salary vs. GDP per capita
- Data from 2024

**Data Visualized:**
- Median/mean salaries in USD
- GDP per capita in USD
- Regional comparisons

### 3. Rich & Poor
Explores income inequality and distribution across countries.

**Features:**
- Customizable axis selection from multiple income metrics
- Scatter plots with trendlines
- 2022 data snapshot

**Data Visualized:**
- Income of poorest 10% (daily, 2017 prices)
- Income of richest 10% (daily, 2017 prices)
- Median income (daily, 2017 prices)
- GDP per capita (PPP, constant 2017 international $)

### 4. Forex
Examines foreign exchange reserves relative to economic development.

**Features:**
- Scatter plot of forex reserves per capita vs. GDP per capita
- 2024 data

**Data Visualized:**
- Forex reserves per person (USD)
- GDP per capita (USD)

### 5. UK Historical GDP
Historical analysis of England's economic development from 1300-1825.

**Features:**
- Two complementary visualizations:
  - GDP per person over time with 20-year moving average
  - GDP per person vs. population (showing the relationship between population growth and prosperity)
- Annotated timeline markers

**Data Visualized:**
- GDP per person in England
- Population of England
- 20-year moving averages

### 6. Bread and Silver
Historical analysis of labor value and cost of living in England (1200-2000).

**Features:**
- Toggle between pence and grams of silver as measurement unit
- Three distinct visualizations:
  - Time required to acquire basic goods (bread/silver) over time
  - Relationship between daily earnings and cost of 2500 kcal of bread
  - Percentage of daily earnings spent on bread
- Includes population overlay

**Data Visualized:**
- Time to acquire 2500 kcal of bread (minutes of labor)
- Time to acquire 1g of silver (minutes of labor)
- Daily earnings (pence and grams of silver)
- Cost of 2500 kcal of bread (pence and grams of silver)
- Population of England

## Installation & Running Locally

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/global-economics-dashboard.git
cd global-economics-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Data Sources

The dashboard uses the following data files (located in the `data/` directory):
- `jobs_vs_gdp_per_capita.csv` - Professional salary data
- `forex_vs_gdp_per_capita.csv` - Foreign exchange reserves data
- `spending_vs_gdp_per_capita.csv` - Government spending data
- `spending_vs_gdp_per_capita_plus_regions.csv` - Regional spending data
- `imf_gross_public_debt_20240924_inverted.csv` - Public debt data
- `daily-income-of-the-poorest-and-richest-decile.csv` - Income distribution data
- `median-daily-per-capita-expenditure-vs-gdp-per-capita.csv` - Median income data
- `uk_historical_gdp.csv` - Historical UK GDP data
- `uk_debt_1692_2023.csv` - Historical UK debt data
- `labour_silver_bread.csv` - Historical labor value data

## Technology Stack

- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computing

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.