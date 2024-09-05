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
    df = pd.read_csv(root_dir_path + "/data/uk_debt_1692_2023.csv").sort_values(
        ["Year"]
    )
    ### Add country column
    df["Country"] = "United Kingdom"
    return df


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
                    "Government Expenditure (IMF & Wiki)": (
                        "Government Expenditure (IMF & Wiki)",
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
                    "Government Expenditure (IMF & Wiki)": "mean",
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
                y="Government Expenditure (IMF & Wiki)",
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
    growth_col = "Average percentage change in GDP per capita USD ({0} - {1})".format(
        growth_range[0], growth_range[1]
    )

    average_spend_df = (
        (
            df.loc[
                df["Year"].isin(list(range(spending_range[0], spending_range[1] + 1))),
                :,
            ]
            .groupby(["Country"])["Government Expenditure (IMF & Wiki)"]
            .mean()
        )
        .reset_index()
        .rename(columns={"Government Expenditure (IMF & Wiki)": spend_col})
    )

    df = pd.merge(
        left=df,
        right=average_spend_df,
        left_on=["Country"],
        right_on=["Country"],
        how="outer",
    )

    df[growth_col] = df.groupby(["Country"])["GDP per capita (OWiD)"].pct_change(
        periods=(growth_range[1] - growth_range[0])
    ) * (100 / (growth_range[1] - growth_range[0]))

    ### Filter to most recent growth range year
    df = df.loc[df["Year"] == growth_range[1]]
    return df, spend_col, growth_col


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
                    label="Remove Countries",
                    options=sorted(all_countries),
                    default=sorted(all_countries),
                    placeholder="Remove countries from plots",
                )
            else:
                remove_countries = st.multiselect(
                    label="Remove Countries",
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
    tab_headers = {"tab1": "Spending & Growth", "tab2": "Salaries", "tab3": "Forex."}
    tab1, tab2, tab3 = st.tabs([tab_headers[k] for k, v in tab_headers.items()])

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
                    debt_adjusted_mode = st.toggle(
                        "Debt Adjusted (UK Only)", value=False
                    )
                long_range = st.slider(
                    "Long-Term Spending and Growth Range",
                    1850,
                    2019,
                    (1999, 2019),
                    help="The range over which multiple 'Spending & Growth' data will be calculated.",
                )
                sub_period = st.number_input(
                    "Spending and Growth Subperiod (years)",
                    value=5,
                    step=1,
                    min_value=1,
                    max_value=169,
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
        y_title_no_brackets = "Average percentage change in GDP per capita USD"
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
                title="Average Government Spending as a Share of GDP vs. Average change in GDP per capita",
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


if __name__ == "__main__":
    main()
