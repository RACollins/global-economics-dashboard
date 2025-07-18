import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import os

root_dir_path = os.path.dirname(os.path.realpath(__file__))


###################
### Page Config ###
###################


st.set_page_config(
    page_title="Global Economics Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="auto",
)


#################
### Constants ###
#################


#################
### Functions ###
#################


@st.cache_data(ttl=3600, show_spinner=False)
def get_jobs_df(root_dir_path):
    df = pd.read_csv(root_dir_path + "/data/jobs_vs_gdp_per_capita.csv")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_forex_df(root_dir_path):
    df = (
        pd.read_csv(root_dir_path + "/data/forex_vs_gdp_per_capita.csv")
        .drop(columns=["Unnamed: 0.1"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_spending_df(root_dir_path):
    df = (
        pd.read_csv(root_dir_path + "/data/spending_vs_gdp_per_capita.csv")
        .drop(columns=["Unnamed: 0"])
        .sort_values(["Country", "Year"])
    )
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_debt_df(root_dir_path):
    df = pd.read_csv(
        root_dir_path + "/data/imf_gross_public_debt_20240924_inverted.csv"
    )

    # Create a complete range of years for each country
    all_years = pd.DataFrame({"Year": range(df["Year"].min(), df["Year"].max() + 1)})
    countries = df["Country"].unique()

    # Create a new dataframe with all combinations of countries and years
    full_df = pd.DataFrame(
        [(country, year) for country in countries for year in all_years["Year"]],
        columns=["Country", "Year"],
    )

    # Merge with original data
    df = pd.merge(full_df, df, on=["Country", "Year"], how="left")

    # Sort the dataframe
    df = df.sort_values(["Country", "Year"])

    # Interpolate missing values for each country
    df["Public debt (% of GDP)"] = df.groupby("Country")[
        "Public debt (% of GDP)"
    ].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))

    return df.sort_values(["Year", "Country"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_uk_historical_gdp_df(root_dir_path):
    df = pd.read_csv(root_dir_path + "/data/uk_historical_gdp.csv")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_uk_historical_labour_df(root_dir_path):
    df = pd.read_csv(root_dir_path + "/data/labour_silver_bread.csv")
    ### Rename relevant columns
    df = df.rename(
        columns={
            "Minutes Req": "2500KCal in Bread",
            "min per g Ag": "1g of Silver",
            "d/day": "Daily Earnings (pence)",
            "g. Ag/day": "Daily Earnings (grams of silver)",
            "bread d/lb": "Price of bread (pence per pound)",
            "bread g. Ag/lb": "Price of bread (grams of silver per pound)",
            "Cost in d for 2500 Kcal": "Cost of 2500KCal of bread (pence)",
            "Cost in Silver for 2500 Kcal Bread": "Cost of 2500KCal of bread (grams of silver)",
        }
    )
    ### Melt to leave only relevant columns
    labour_melted_df = df.melt(
        id_vars=["Year"],
        value_vars=[
            "2500KCal in Bread",
            "1g of Silver",
        ],
        var_name="Labour measure",
        value_name="Time to aquire (minutes)",
    )
    pence_melted_df = df.melt(
        id_vars=["Year"],
        value_vars=[
            "Cost of 2500KCal of bread (pence)",
            "Daily Earnings (pence)",
        ],
        var_name="Measure (pence)",
        value_name="Pence",
    )
    pence_melted_df["Measure (pence)"] = pence_melted_df["Measure (pence)"].replace(
        {
            "Cost of 2500KCal of bread (pence)": "2500KCal of bread",
            "Daily Earnings (pence)": "Daily Earnings",
        }
    )
    silver_melted_df = df.melt(
        id_vars=["Year"],
        value_vars=[
            "Cost of 2500KCal of bread (grams of silver)",
            "Daily Earnings (grams of silver)",
        ],
        var_name="Measure (grams of silver)",
        value_name="Grams of silver",
    )
    silver_melted_df["Measure (grams of silver)"] = silver_melted_df[
        "Measure (grams of silver)"
    ].replace(
        {
            "Cost of 2500KCal of bread (grams of silver)": "2500KCal of bread",
            "Daily Earnings (grams of silver)": "Daily Earnings",
        }
    )
    ### Merge in pence data
    final_df = pd.merge(
        left=labour_melted_df,
        right=pence_melted_df,
        left_on=["Year"],
        right_on=["Year"],
    )
    ### Merge in silver data
    final_df = pd.merge(
        left=final_df,
        right=silver_melted_df,
        left_on=["Year"],
        right_on=["Year"],
    )
    ### Merge in population
    final_df = pd.merge(
        left=final_df,
        right=df.loc[:, ["Year", "Population (England)"]],
        left_on="Year",
        right_on="Year",
    )
    return final_df.sort_values(["Year"]).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def make_region_avg_df(spending_df, weight_pop):
    if weight_pop:
        wm = lambda x: np.average(x, weights=spending_df.loc[x.index, "Population"])
        region_avg_spending_df = (
            spending_df.groupby(["Region", "Year"])
            .agg(
                **{
                    "Population": ("Population", "sum"),
                    "GDP per capita (OWiD)": ("GDP per capita (OWiD)", wm),
                    "Government Expenditure (IMF, Wiki, Statistica)": (
                        "Government Expenditure (IMF, Wiki, Statistica)",
                        wm,
                    ),
                }
            )
            .reset_index()
        )
    else:
        region_avg_spending_df = (
            spending_df.groupby(["Region", "Year"])
            .agg(
                {
                    "Population": "sum",
                    "GDP per capita (OWiD)": "mean",
                    "Government Expenditure (IMF, Wiki, Statistica)": "mean",
                }
            )
            .reset_index()
        )
    region_avg_spending_df["Country"] = region_avg_spending_df["Region"] + "_avg"
    return region_avg_spending_df


@st.cache_data(ttl=3600, show_spinner=False)
def make_line_plots(df):
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for country in df["Country"].unique():
        filtered_df = df.loc[df["Country"] == country, :]
        fig.add_trace(
            px.line(
                filtered_df,
                x="Year",
                y="GDP per capita (OWiD)",
                color="Region",
                color_discrete_sequence=[
                    "red",
                    "magenta",
                    "goldenrod",
                    "green",
                    "blue",
                ],
                category_orders={
                    "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
                },
                hover_data={
                    "Country": True,
                    "Population": True,
                },
            ).data[0],
            row=1,
            col=1,
        )
        fig.add_trace(
            px.line(
                filtered_df,
                x="Year",
                y="Government Expenditure (IMF, Wiki, Statistica)",
                color="Region",
                color_discrete_sequence=[
                    "red",
                    "magenta",
                    "goldenrod",
                    "green",
                    "blue",
                ],
                category_orders={
                    "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
                },
                hover_data={
                    "Country": True,
                    "Population": True,
                },
            ).data[0],
            row=2,
            col=1,
        )
    seen_regions = set()
    fig.for_each_trace(
        lambda trace: (
            trace.update(showlegend=False)
            if (trace.name in seen_regions)
            else seen_regions.add(trace.name)
        )
    )
    return fig


def apply_graph_stylings(fig):
    fig.update_traces(textposition="top center")
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )
    return fig


def add_repeats(df, lowest_repeat, highest_repeat):
    df["repeat"] = (
        (df["Population"] - df["Population"].min())
        / (df["Population"].max() - df["Population"].min())
        * (highest_repeat - lowest_repeat)
        + lowest_repeat
    ).round()
    df = df.loc[df.index.repeat(df["repeat"])]
    return df


def add_debt_adjustment(df, debt_df):
    df = pd.merge(
        left=df,
        right=debt_df.loc[:, ["Year", "Country", "Public debt (% of GDP)"]],
        left_on=["Year", "Country"],
        right_on=["Year", "Country"],
        how="left",
    )
    ### Calculate total debt and difference year on year
    df["Debt per capita"] = df["GDP per capita (OWiD)"] * (
        df["Public debt (% of GDP)"] / 100
    )
    ### Calculate difference of total debt year on year
    df["Debt change per capita"] = df.groupby(["Country"])["Debt per capita"].diff()
    ### Redefine GDP per capita with debt adjustment
    df["GDP per capita (OWiD)"] = (
        df["GDP per capita (OWiD)"] - df["Debt per capita"] * 0.1
    )
    return df


def add_country_lables(fig, df, countries, x_title, y_title, log_x, log_y):
    for country in countries:
        x_array = df.loc[df["Country"] == country, x_title].values
        y_array = df.loc[df["Country"] == country, y_title].values
        if x_array.size == 0 or y_array.size == 0:
            continue
        x, y = x_array[0], y_array[0]
        x = np.log10(x) if log_x else x
        y = np.log10(y) if log_y else y
        fig.add_annotation(x=x, y=y, text=country, showarrow=True)
    return fig


def transform_spending_df(df, spending_range, growth_range):
    spend_col = "Average Government Expenditure as % of GDP ({0} - {1})".format(
        spending_range[0], spending_range[1]
    )
    growth_col = (
        "Annualized percentage change in GDP per capita USD ({0} - {1})".format(
            growth_range[0], growth_range[1]
        )
    )

    average_spend_df = (
        (
            df.loc[
                df["Year"].isin(list(range(spending_range[0], spending_range[1] + 1))),
                :,
            ]
            .groupby(["Country"])["Government Expenditure (IMF, Wiki, Statistica)"]
            .mean()
        )
        .reset_index()
        .rename(columns={"Government Expenditure (IMF, Wiki, Statistica)": spend_col})
    )

    df = pd.merge(
        left=df,
        right=average_spend_df,
        left_on=["Country"],
        right_on=["Country"],
        how="outer",
    )

    # Fill NA values before calculating growth rate
    df["GDP per capita (OWiD)"] = df.groupby("Country")["GDP per capita (OWiD)"].ffill()

    # Calculate annualized growth rate
    # First get start and end GDP values for each country, ensuring unique values
    start_gdp = (
        df[df["Year"] == growth_range[0]]
        .groupby("Country")["GDP per capita (OWiD)"]
        .first()
    )
    end_gdp = (
        df[df["Year"] == growth_range[1]]
        .groupby("Country")["GDP per capita (OWiD)"]
        .first()
    )

    # Calculate annualized growth rate: (end/start)^(1/x) - 1
    x = growth_range[1] - growth_range[0]  # number of years
    growth_rates = ((end_gdp / start_gdp) ** (1 / x) - 1) * 100  # convert to percentage

    # Add growth rates back to dataframe
    df[growth_col] = df["Country"].map(growth_rates)

    ### Filter to most recent growth range year
    df = df.loc[df["Year"] == growth_range[1]]
    return df, spend_col, growth_col


@st.cache_data(ttl=3600, show_spinner=False)
def get_rich_poor_df(root_dir_path):
    # Load both datasets
    poor_rich = pd.read_csv(
        root_dir_path + "/data/daily-income-of-the-poorest-and-richest-decile.csv"
    )
    median_gdp = pd.read_csv(
        root_dir_path
        + "/data/median-daily-per-capita-expenditure-vs-gdp-per-capita.csv"
    )

    # Filter for 2022 only
    poor_rich_2022 = poor_rich[poor_rich["Year"] == 2022].copy()
    median_gdp_2022 = median_gdp[median_gdp["Year"] == 2022].copy()

    # Merge on country code (Code)
    merged = pd.merge(
        poor_rich_2022,
        median_gdp_2022,
        left_on="Code",
        right_on="Code",
        suffixes=("_poor_rich", "_median_gdp"),
    )

    # Rename columns for clarity and axis selection
    merged = merged.rename(
        columns={
            "Entity_poor_rich": "Country",
            "World regions according to OWID_poor_rich": "Region",
            "Population (historical)_poor_rich": "Population",
            "Threshold (Poorest decile, 2017 prices) - Income or consumption consolidated": "Income of poorest 10%",
            "Threshold (Richest decile, 2017 prices) - Income or consumption consolidated": "Income of richest 10%",
            "Median (2017 prices) - Income or consumption consolidated": "Median income",
            "GDP per capita, PPP (constant 2017 international $)": "GDP per capita",
        }
    )

    # Select only relevant columns
    merged = merged[
        [
            "Country",
            "Region",
            "Population",
            "Income of poorest 10%",
            "Income of richest 10%",
            "Median income",
            "GDP per capita",
        ]
    ].copy()

    # Convert to numeric
    for col in [
        "Income of poorest 10%",
        "Income of richest 10%",
        "Median income",
        "GDP per capita",
        "Population",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Drop rows with missing values in any of the main columns
    merged = merged.dropna(
        subset=[
            "Income of poorest 10%",
            "Income of richest 10%",
            "Median income",
            "GDP per capita",
            "Population",
        ]
    )

    # Map North America and South America to Americas for consistency
    merged["Region"] = merged["Region"].replace(
        {
            "North America": "Americas",
            "South America": "Americas",
        }
    )

    return merged


##################
### App proper ###
##################


def main():
    st.title("Global Economics Dashboard")
    st.subheader("View various global economic data")

    ### Import data
    jobs_df = get_jobs_df(root_dir_path)
    forex_df = get_forex_df(root_dir_path).astype({"GDP_per_capita_USD": "float64"})
    spending_df = get_spending_df(root_dir_path)
    all_countries = (
        pd.concat([spending_df["Country"], forex_df["Country"]])
        .drop_duplicates()
        .to_list()
    )

    ### Side bar
    with st.sidebar:
        st.header("Plot Options")
        with st.container(border=True):
            st.markdown("##### Axes")
            log_x = st.checkbox("log_x")
            log_y = st.checkbox("log_y")
        with st.container(border=True):
            st.markdown("##### Population")
            show_pop = st.checkbox(
                "Display",
                help="Display the population of each country as marker size.",
            )
            weight_pop = st.checkbox(
                "Weight",
                help="Weight trendlines by the population of each country.",
            )
        with st.container(border=True):
            display_countries = st.multiselect(
                label="Country Labels",
                options=sorted(all_countries),
                placeholder="Add country labels to plots",
            )
        with st.container(border=True):
            remove_all_countries = st.checkbox("Remove All Countries")

            if remove_all_countries:
                remove_countries = st.multiselect(
                    label="Removed Countries",
                    options=sorted(all_countries),
                    default=sorted(all_countries),
                    placeholder="Remove countries from plots",
                )
            else:
                remove_countries = st.multiselect(
                    label="Removed Countries",
                    options=sorted(all_countries),
                    placeholder="Remove countries from plots",
                )

    if remove_countries:
        jobs_df = jobs_df.loc[~jobs_df["Country"].isin(remove_countries), :]
        forex_df = forex_df.loc[~forex_df["Country"].isin(remove_countries), :]
        spending_df = spending_df.loc[~spending_df["Country"].isin(remove_countries), :]
    if weight_pop:
        lowest_repeat, highest_repeat = 1, 100

    ### Tabs
    tab_headers = {
        "tab1": "Spending & Growth",
        "tab2": "Salaries",
        "tab3": "Rich & Poor",
        "tab4": "Forex.",
        "tab5": "UK Historical GDP",
        "tab6": "Bread and Silver",
    }
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [tab_headers[k] for k, v in tab_headers.items()]
    )

    with tab1:
        ### Bottom Filters
        btm_left_years_buffer, btm_centre_years_col, btm_right_years_buffer = (
            st.columns([1, 10, 1])
        )
        with btm_centre_years_col:
            toggle_col1, toggle_col2, toggle_buffer = st.columns([1, 1, 1])
            with st.container(border=True):
                with toggle_col1:
                    region_avg_mode = st.toggle("Region Averages", value=False)
                with toggle_col2:
                    debt_adjusted_mode = st.toggle("Debt Adjusted", value=False)
                long_range = st.slider(
                    "Long-Term Spending and Growth Range",
                    1850,
                    2022,
                    (1999, 2022),
                    help="The range over which multiple 'Spending & Growth' data will be calculated.",
                )
                sub_period = st.number_input(
                    "Spending and Growth Subperiod (years)",
                    value=5,
                    step=1,
                    min_value=1,
                    max_value=172,
                    help="The length of time over which a single 'Spending & Growth' datum will be calculated.",
                )
                nPeriods = long_range[1] - (long_range[0] + sub_period) + 1
                if nPeriods < 0:
                    st.error(
                        body="'Subperiod' is larger than 'Long-Term Spending and Growth Range'",
                        icon="⚠️",
                    )
                else:
                    st.write("Number of Subperiods: {}".format(nPeriods))

        ### Display line graphs
        if debt_adjusted_mode:
            debt_df = get_debt_df(root_dir_path)
            spending_df = add_debt_adjustment(spending_df, debt_df)
        region_avg_df = make_region_avg_df(spending_df, weight_pop)
        if region_avg_mode:
            plot_spending_df = region_avg_df
        else:
            plot_spending_df = spending_df

        fig = make_line_plots(df=plot_spending_df)
        fig.update_traces(
            line=dict(
                width=1.0,
            )
        )
        fig.update_xaxes(type="log" if log_x else "linear", row=1, col=1)
        fig.update_xaxes(
            title_text="Year", type="log" if log_x else "linear", row=2, col=1
        )
        fig.update_yaxes(
            title_text="GDP per capita",
            type="log" if log_y else "linear",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Government Expenditure",
            type="log" if log_y else "linear",
            row=2,
            col=1,
        )
        fig.update_layout(height=600, width=600)
        fig.add_vline(
            x=long_range[0],
            line_width=1.5,
            line_dash="dash",
            line_color="red",
        )
        fig.add_vline(
            x=long_range[1],
            line_width=1.5,
            line_dash="dash",
            line_color="red",
        )

        fig = apply_graph_stylings(fig)
        with st.container(border=True):
            st.plotly_chart(fig, theme=None, use_container_width=True)
            ### Download as CSV
            dwnld_csv_btn = st.download_button(
                label="Download as CSV",
                data=plot_spending_df.to_csv(index=True, header=True).encode("utf-8"),
                file_name="GDP_per_capita_Government_Expenditure_1850_2020.csv",
                mime="text/csv",
            )

        ### Generate "scatter" data
        x_title_no_brackets = "Average Government Expenditure as % of GDP"
        y_title_no_brackets = "Annualized percentage change in GDP per capita USD"
        all_subperiod_df_list = []
        for p in range(nPeriods):
            sg_range = (long_range[0] + p, long_range[0] + p + sub_period)
            subperiod_df, spend_col, growth_col = transform_spending_df(
                df=plot_spending_df, spending_range=sg_range, growth_range=sg_range
            )
            subperiod_df = subperiod_df.loc[
                :, ["Country", "Region", "Population", spend_col, growth_col]
            ].rename(
                columns={
                    spend_col: x_title_no_brackets,
                    growth_col: y_title_no_brackets,
                }
            )
            subperiod_df["start_year"] = sg_range[0]
            subperiod_df["end_year"] = sg_range[1]
            all_subperiod_df_list.append(subperiod_df)
        all_subperiod_df = pd.concat(all_subperiod_df_list).reset_index(drop=True)

        selected_plot = st.selectbox(
            label="Plot Type",
            options=["Scatter", "Heatmap"],
            index=0,
        )

        ### Plot scatter/heatmap
        if selected_plot == "Scatter":
            ### Add repeats
            if weight_pop and not region_avg_mode:
                all_subperiod_df = add_repeats(
                    all_subperiod_df, lowest_repeat, highest_repeat
                )

            ### Plot scatter
            size = "Population" if show_pop else None
            fig = px.scatter(
                all_subperiod_df,
                x=x_title_no_brackets,
                y=y_title_no_brackets,
                color="Region",
                color_discrete_sequence=[
                    "red",
                    "magenta",
                    "goldenrod",
                    "green",
                    "blue",
                ],
                category_orders={
                    "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
                },
                size=size,
                size_max=80,
                opacity=0.2,
                hover_data={
                    "Country": True,
                    "Population": True,
                    "start_year": True,
                    "end_year": True,
                },
                trendline="ols",
                trendline_scope="overall",
                trendline_color_override="black",
                title="Average Government Spending as a Share of GDP vs. Annualized change in GDP per capita",
                log_x=log_x,
                log_y=log_y,
            )
            fig = apply_graph_stylings(fig)
            fig = add_country_lables(
                fig,
                df=all_subperiod_df,
                countries=display_countries,
                x_title=x_title_no_brackets,
                y_title=y_title_no_brackets,
                log_x=log_x,
                log_y=log_y,
            )
            with st.container(border=True):
                st.plotly_chart(fig, theme=None, use_container_width=True)
                ### Download as CSV
                dwnld_csv_btn = st.download_button(
                    label="Download as CSV",
                    data=all_subperiod_df.to_csv(index=True, header=True).encode(
                        "utf-8"
                    ),
                    file_name="{0}_vs_{1}.csv".format(
                        x_title_no_brackets, y_title_no_brackets
                    ),
                    mime="text/csv",
                )
        elif selected_plot == "Heatmap":
            if weight_pop:
                fig = px.density_heatmap(
                    all_subperiod_df,
                    x=x_title_no_brackets,
                    y=y_title_no_brackets,
                    z="Population",
                    range_x=[0, 101],
                    histfunc="sum",
                    color_continuous_scale="Hot_r",
                )
            else:
                fig = px.density_heatmap(
                    all_subperiod_df,
                    x=x_title_no_brackets,
                    y=y_title_no_brackets,
                    range_x=[0, 101],
                    color_continuous_scale="Hot_r",
                )
            ### May implement bin sizes
            # fig.update_traces(
            #    xbins=dict(start=0.0, end=100.0, size=5),
            #    ybins=dict(start=-60.0, end=100.0, size=5),
            # )
            fig = add_country_lables(
                fig,
                df=all_subperiod_df,
                countries=display_countries,
                x_title=x_title_no_brackets,
                y_title=y_title_no_brackets,
                log_x=log_x,
                log_y=log_y,
            )
            with st.container(border=True):
                st.plotly_chart(fig, theme=None, use_container_width=True)
                ### Download as CSV
                dwnld_csv_btn = st.download_button(
                    label="Download as CSV",
                    data=all_subperiod_df.to_csv(index=True, header=True).encode(
                        "utf-8"
                    ),
                    file_name="{0}_vs_{1}.csv".format(
                        x_title_no_brackets, y_title_no_brackets
                    ),
                    mime="text/csv",
                )

    with tab2:
        ### Filters
        left_jobs_buffer, centre_jobs_col, right_jobs_buffer = st.columns([2, 8, 2])
        with centre_jobs_col:
            with st.container(border=True):
                selected_year = 2024
                selected_job = st.selectbox(
                    label="Job",
                    options=["Bricklayer", "Doctor", "Nurse", "All"],
                    index=2,
                )

        ### Apply filters and repeats
        job_df = (
            jobs_df.loc[
                (jobs_df["Job"] == selected_job) & (jobs_df["Year"] == selected_year),
                :,
            ]
            .drop(columns=["Unnamed: 0"])
            .drop_duplicates()
            .reset_index(drop=True)
        ).astype({"GDP_per_capita_USD": "float64"})

        if weight_pop:
            job_df = add_repeats(job_df, lowest_repeat, highest_repeat)

        ### Plot
        size = "Population" if show_pop else None
        x_title, y_title = "GDP_per_capita_USD", "Median_USD"
        y_title = "Mean_USD" if selected_job == "All" else "Median_USD"
        fig = px.scatter(
            job_df,
            x=x_title,
            y=y_title,
            color="Region",
            color_discrete_sequence=["red", "magenta", "goldenrod", "green", "blue"],
            category_orders={
                "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
            },
            size=size,
            size_max=80,
            hover_data={"Country": True, "Population": True},
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="black",
            title="{0} pay of {1} VS. GDP per capita ({2})".format(
                y_title.split("_")[0], selected_job, selected_year
            ),
            log_x=log_x,
            log_y=log_y,
        )
        fig = apply_graph_stylings(fig)
        fig = add_country_lables(
            fig,
            df=job_df,
            countries=display_countries,
            x_title=x_title,
            y_title=y_title,
            log_x=log_x,
            log_y=log_y,
        )
        with st.container(border=True):
            st.plotly_chart(fig, theme=None, use_container_width=True)
            ### Download as CSV
            dwnld_csv_btn = st.download_button(
                label="Download as CSV",
                data=job_df.loc[
                    :, ["Country", "Region", "Population", x_title, y_title]
                ]
                .to_csv(index=True, header=True)
                .encode("utf-8"),
                file_name="{0}_vs_{1}.csv".format(x_title, y_title),
                mime="text/csv",
            )

    with tab3:
        rich_poor_df = get_rich_poor_df(root_dir_path)

        axis_options = [
            "Income of poorest 10%",
            "Income of richest 10%",
            "Median income",
            "GDP per capita",
        ]
        left_rp_buffer, centre_rp_col, right_rp_buffer = st.columns([2, 8, 2])
        with centre_rp_col:
            with st.container(border=True):
                x_axis = st.selectbox(
                    label="X Axis",
                    options=axis_options,
                    index=2,  # Default: Median income
                )
                y_axis = st.selectbox(
                    label="Y Axis",
                    options=axis_options,
                    index=0,  # Default: Income of poorest 10%
                )

        # Apply population weighting if selected
        plot_df = rich_poor_df.copy()
        if remove_countries:
            plot_df = plot_df.loc[~plot_df["Country"].isin(remove_countries), :]
        if weight_pop:
            plot_df = add_repeats(plot_df, lowest_repeat, highest_repeat)
        size = "Population" if show_pop else None

        # Dynamic plot title
        plot_title = f"{y_axis} vs. {x_axis} (2022)"

        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color="Region",
            color_discrete_sequence=["red", "magenta", "goldenrod", "green", "blue"],
            category_orders={
                "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
            },
            size=size,
            size_max=80,
            hover_data={"Country": True, "Population": True},
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="black",
            title=plot_title,
            log_x=log_x,
            log_y=log_y,
        )
        fig = apply_graph_stylings(fig)
        fig = add_country_lables(
            fig,
            df=plot_df,
            countries=display_countries,
            x_title=x_axis,
            y_title=y_axis,
            log_x=log_x,
            log_y=log_y,
        )
        with st.container(border=True):
            st.plotly_chart(fig, theme=None, use_container_width=True)
            dwnld_csv_btn = st.download_button(
                label="Download as CSV",
                data=plot_df.loc[:, ["Country", "Region", "Population", x_axis, y_axis]]
                .to_csv(index=True, header=True)
                .encode("utf-8"),
                file_name=f"{x_axis}_vs_{y_axis}_rich_poor_2022.csv",
                mime="text/csv",
            )

    with tab4:
        ### Apply filters and repeats
        if weight_pop:
            forex_df = add_repeats(forex_df, lowest_repeat, highest_repeat)

        ### Plot
        size = "Population" if show_pop else None
        x_title, y_title = "GDP_per_capita_USD", "Forex_Reserves_per_person_USD"
        fig = px.scatter(
            forex_df,
            x=x_title,
            y=y_title,
            color="Region",
            color_discrete_sequence=["red", "magenta", "goldenrod", "green", "blue"],
            category_orders={
                "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
            },
            size=size,
            size_max=80,
            hover_data={"Country": True, "Population": True},
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="black",
            title="Forex Reserves Including Gold per Person VS. GDP per capita (2024)",
            log_x=log_x,
            log_y=log_y,
        )
        fig = apply_graph_stylings(fig)
        fig = add_country_lables(
            fig,
            df=forex_df,
            countries=display_countries,
            x_title=x_title,
            y_title=y_title,
            log_x=log_x,
            log_y=log_y,
        )
        with st.container(border=True):
            st.plotly_chart(fig, theme=None, use_container_width=True)
            ### Download as CSV
            dwnld_csv_btn = st.download_button(
                label="Download as CSV",
                data=forex_df.loc[
                    :, ["Country", "Region", "Population", x_title, y_title]
                ]
                .to_csv(index=True, header=True)
                .encode("utf-8"),
                file_name="{0}_vs_{1}.csv".format(x_title, y_title),
                mime="text/csv",
            )

    with tab5:
        uk_historical_gdp_df = get_uk_historical_gdp_df(root_dir_path)

        ### Filter df
        uk_historical_gdp_df = uk_historical_gdp_df.loc[
            uk_historical_gdp_df["Year"].between(1300, 1825)
        ].reset_index(drop=True)

        ### Plot
        size = "Population (England)" if show_pop else None
        x_title, y_title = "Year", "GDP Per Person"
        upper_fig = px.scatter(
            uk_historical_gdp_df,
            x=x_title,
            y=y_title,
            color="Region",
            color_discrete_sequence=["red", "magenta", "goldenrod", "green", "blue"],
            category_orders={
                "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
            },
            size=size,
            size_max=80,
            opacity=0.35,
            hover_data={"Country": True, "Population (England)": True},
            title="Historical GDP Per Person in England",
            log_x=log_x,
            log_y=log_y,
        )
        ### Add moving average line
        upper_fig.add_trace(
            px.line(
                uk_historical_gdp_df,
                x="Year",
                y="GDP Per Person (20-year moving average)",
                title="20-year moving average",
                color_discrete_sequence=["black"],
            ).data[0]
        )
        upper_fig = apply_graph_stylings(upper_fig)

        ### Plot with population as x-axis and GDP per person as y-axis
        x_title, y_title = "Population (England)", "GDP Per Person"
        lower_fig = px.scatter(
            uk_historical_gdp_df,
            x=x_title,
            y=y_title,
            color="Region",
            color_discrete_sequence=["red", "magenta", "goldenrod", "green", "blue"],
            category_orders={
                "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
            },
            size=size,
            size_max=80,
            opacity=0.35,
            hover_data={"Country": True, "Population (England)": True, "Year": True},
            title="GDP Per Person vs. Population in England",
            log_x=log_x,
            log_y=log_y,
        )
        ### Add moving average line
        lower_fig.add_trace(
            px.line(
                uk_historical_gdp_df,
                x="Population (England)",
                y="GDP Per Person (20-year moving average)",
                hover_data={"Year": True},
                title="20-year moving average",
                color_discrete_sequence=["black"],
            ).data[0]
        )
        ### Add year annotations at regular intervals
        df_subset = uk_historical_gdp_df.loc[
            uk_historical_gdp_df["Year"].isin(list(range(1300, 1825, 50)) + [1825])
        ]

        for _, row in df_subset.iterrows():
            lower_fig.add_annotation(
                x=row["Population (England)"],
                y=row["GDP Per Person (20-year moving average)"],
                text=str(int(row["Year"])),
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(size=10),
            )
        lower_fig = apply_graph_stylings(lower_fig)
        with st.container(border=True):
            st.plotly_chart(upper_fig, theme=None, use_container_width=True)
            st.plotly_chart(lower_fig, theme=None, use_container_width=True)
            ### Download as CSV
            dwnld_csv_btn = st.download_button(
                label="Download as CSV",
                data=uk_historical_gdp_df.to_csv(index=True, header=True).encode(
                    "utf-8"
                ),
                file_name="GDP_Per_Person_in_England_1300_1825.csv",
                mime="text/csv",
            )

    with tab6:
        labour_df = get_uk_historical_labour_df(root_dir_path)
        ### Get unique population data for each year
        population_df = labour_df[["Year", "Population (England)"]].drop_duplicates()
        ### Select pence or silver
        which_measure = st.radio(
            "Pence or Silver?",
            ["Pence", "Grams of silver"],
            key="which_measure_radio",
            label_visibility="visible",
            horizontal=True,
        )

        ### Plot silver/pence data
        with st.container(border=True):
            colour = (
                "Measure (pence)"
                if which_measure == "Pence"
                else "Measure (grams of silver)"
            )
            fig = px.line(
                labour_df,
                title="Give us today our daily bread...",
                x="Year",
                y=which_measure,
                log_x=log_x,
                log_y=log_y,
                color=colour,
                color_discrete_sequence=px.colors.qualitative.Plotly,
                custom_data=["Year"],
            )

            ### Update layout to include secondary y-axis with log scale and unified hover
            fig.update_layout(
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.2,
                ),
            )

            ### Update hover template for the time traces
            fig.update_traces(
                hovertemplate="%{y:.0f} minutes<br>%{x}<extra>%{fullData.name}</extra>",
                selector=dict(yaxis="y1"),
            )

            fig = apply_graph_stylings(fig)
            st.plotly_chart(fig, theme=None, use_container_width=True)
            ### Download as CSV
            dwnld_csv_btn = st.download_button(
                label="Download as CSV",
                data=labour_df.to_csv(index=True, header=True).encode("utf-8"),
                file_name="Bread_cost_in_England_1200_2000.csv",
                mime="text/csv",
            )

        ### Earnings vs. cost of 2500KCal of bread over time
        ### Filter then pivot first
        pivoted_labour_df = (
            labour_df.loc[
                :,
                [
                    "Year",
                    "Measure ({})".format(which_measure.lower()),
                    "{}".format(which_measure),
                ],
            ]
            .drop_duplicates()
            .reset_index(drop=True)
            .pivot(
                index="Year",
                columns="Measure ({})".format(which_measure.lower()),
                values="{}".format(which_measure),
            )
            .reset_index()
        )
        pivoted_labour_df["% of daily earnings spent on bread"] = (
            pivoted_labour_df["2500KCal of bread"] / pivoted_labour_df["Daily Earnings"]
        ) * 100

        fig_xaxis_not_time = px.line(
            pivoted_labour_df,
            x="Daily Earnings",
            y="2500KCal of bread",
            # color="Measure (pence)",
            hover_data={
                "Daily Earnings": True,
                "2500KCal of bread": True,
                "Year": True,
            },
            title=f"Daily earnings vs. cost of 2500KCal of bread ({which_measure})",
            log_x=log_x,
            log_y=log_y,
        )
        fig_xaxis_not_time = apply_graph_stylings(fig_xaxis_not_time)

        ### Percentage of daily earnings spend on bread
        fig_xaxis_time = px.line(
            pivoted_labour_df,
            x="Year",
            y="% of daily earnings spent on bread",
            # color="Measure (pence)",
            hover_data={
                "Daily Earnings": True,
                "2500KCal of bread": True,
                "% of daily earnings spent on bread": True,
                "Year": True,
            },
            title=f"Percentage of daily earnings spent on bread ({which_measure})",
            log_x=log_x,
            log_y=log_y,
        )
        fig_xaxis_time = apply_graph_stylings(fig_xaxis_time)
        with st.container(border=True):
            st.plotly_chart(fig_xaxis_not_time, theme=None, use_container_width=True)
            st.plotly_chart(fig_xaxis_time, theme=None, use_container_width=True)
            ### Download as CSV
            dwnld_csv_btn = st.download_button(
                label="Download as CSV",
                data=pivoted_labour_df.to_csv(index=True, header=True).encode("utf-8"),
                file_name="Daily_earnings_and_cost_of_2500KCal_of_bread_1300_1825.csv",
                mime="text/csv",
            )

        ### Plot labour data
        x_title, y_title = "Year", "Time to aquire (minutes)"
        fig = px.line(
            labour_df,
            title="Value of Labour of an Agricultural Worker in England (1200 to 2000)",
            x=x_title,
            y=y_title,
            log_x=log_x,
            log_y=log_y,
            color="Labour measure",
            custom_data=["Year"],
        )

        ### Add population trace with secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=population_df["Year"],
                y=population_df["Population (England)"],
                name="Population",
                line=dict(color="black", dash="dot"),
                yaxis="y2",
                hovertemplate="Population: %{y:,.0f}<br>Year: %{x}<extra></extra>",
            )
        )

        ### Create tick values and text with "M" only on 1M, 10M, 100M
        tick_vals = [
            1e6,
            2e6,
            3e6,
            4e6,
            5e6,
            6e6,
            7e6,
            8e6,
            9e6,
            1e7,
            2e7,
            3e7,
            4e7,
            5e7,
            6e7,
            7e7,
            8e7,
            9e7,
            1e8,
        ]
        tick_text = [
            f"{int(val/1e6)}M" if val in [1e6, 1e7, 1e8] else f"{int(val/1e6)}"
            for val in tick_vals
        ]

        ### Update layout to include secondary y-axis with log scale and unified hover
        fig.update_layout(
            yaxis2=dict(
                title="Population",
                overlaying="y",
                side="right",
                showgrid=False,
                type="log",
                range=[6, 8],  # 10^6 (1M) to 10^8 (100M)
                tickvals=tick_vals,
                ticktext=tick_text,
            ),
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.2,
            ),
        )

        ### Update hover template for the time traces
        fig.update_traces(
            hovertemplate="%{y:.0f} minutes<br>%{x}<extra>%{fullData.name}</extra>",
            selector=dict(yaxis="y1"),
        )

        fig = apply_graph_stylings(fig)
        with st.container(border=True):
            st.plotly_chart(fig, theme=None, use_container_width=True)
            ### Download as CSV
            dwnld_csv_btn = st.download_button(
                label="Download as CSV",
                data=labour_df.to_csv(index=True, header=True).encode("utf-8"),
                file_name="Value_of_Labour_of_an_Agricultural_Worker_in_England_1200_2000.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
