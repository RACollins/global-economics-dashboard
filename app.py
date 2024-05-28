import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
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
    df = pd.read_csv(root_dir_path + "/data/spending_vs_gdp_per_capita.csv").drop(
        columns=["Unnamed: 0"]
    )
    return df


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
    growth_col = "Percentage change in GDP per capita USD ({0} - {1})".format(
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

    df[growth_col] = (
        df.groupby(["Country"])["GDP per capita (OWiD)"].pct_change(
            periods=(growth_range[1] - growth_range[0])
        )
        * 100
    )

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
    tab_headers = {"tab1": "Salaries", "tab2": "Forex.", "tab3": "Spending & Growth"}
    tab1, tab2, tab3 = st.tabs([tab_headers[k] for k, v in tab_headers.items()])

    with tab1:
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
    with tab2:
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
    with tab3:
        ### Top Filters
        top_left_years_buffer, top_centre_years_col, top_right_years_buffer = (
            st.columns([1, 10, 1])
        )
        with top_centre_years_col:
            with st.container(border=True):
                spending_range = st.slider(
                    "Spending Range",
                    1850,
                    2019,
                    (2003, 2011),
                    help="The average government expenditure (as a percentage of GDP) over the given time period.",
                )
                growth_range = st.slider(
                    "Growth Range",
                    1850,
                    2019,
                    (2003, 2011),
                    help="The percentage change in GDP per capita over the given time period.",
                )

        ### Apply filters and transform
        transformed_spending_df, spend_col, growth_col = transform_spending_df(
            df=spending_df, spending_range=spending_range, growth_range=growth_range
        )

        ### Add repeats
        if weight_pop:
            transformed_spending_df = add_repeats(
                transformed_spending_df, lowest_repeat, highest_repeat
            )

        ### Plot
        size = "Population" if show_pop else None
        x_title, y_title = spend_col, growth_col
        fig = px.scatter(
            transformed_spending_df,
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
            title="Average Government Spending as a Share of GDP vs. Change in GDP per capita",
            log_x=log_x,
            log_y=log_y,
        )
        fig = apply_graph_stylings(fig)
        fig = add_country_lables(
            fig,
            df=transformed_spending_df,
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
                data=transformed_spending_df.loc[
                    :, ["Country", "Region", "Population", x_title, y_title]
                ]
                .to_csv(index=True, header=True)
                .encode("utf-8"),
                file_name="{0}_vs_{1}.csv".format(x_title, y_title),
                mime="text/csv",
            )

        st.divider()

        ### Bottom Filters
        btm_left_years_buffer, btm_centre_years_col, btm_right_years_buffer = (
            st.columns([1, 10, 1])
        )
        with btm_centre_years_col:
            with st.container(border=True):
                long_range = st.slider(
                    "Long-Term Spending and Growth Range",
                    1850,
                    2019,
                    (1900, 2011),
                    help="Placeholder",
                )
                sub_period = st.number_input(
                    "Spending and Growth Subperiod",
                    value=10,
                    step=1,
                    min_value=1,
                    max_value=25,
                    help="Placeholder",
                )
                nPeriods = long_range[1] - (long_range[0] + sub_period) + 1
                if nPeriods < 0:
                    st.error(
                        body="Please decrease 'Subperiod' or increase 'Long-Term Spending and Growth Range'",
                        icon="⚠️",
                    )
                else:
                    st.write("Number of Subperiods: {}".format(nPeriods))

        ### Generate "scatter" data
        x_title_no_brackets = "Average Government Expenditure as % of GDP"
        y_title_no_brackets = "Percentage change in GDP per capita USD"
        all_subperiod_df_list = []
        for p in range(nPeriods):
            sg_range = (long_range[0] + p, long_range[0] + p + sub_period)
            subperiod_df, spend_col, growth_col = transform_spending_df(
                df=spending_df, spending_range=sg_range, growth_range=sg_range
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
        all_subperiod_scatter_df = pd.concat(all_subperiod_df_list).reset_index(
            drop=True
        )

        ### Generate "Heatmap" data

        selected_plot = st.selectbox(
            label="Plot Type",
            options=["Scatter", "Heatmap"],
            index=0,
        )

        if selected_plot == "Scatter":
            ### Add repeats
            if weight_pop:
                all_subperiod_scatter_df = add_repeats(
                    all_subperiod_scatter_df, lowest_repeat, highest_repeat
                )

            ### Plot scatter
            fig = px.scatter(
                all_subperiod_scatter_df,
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
                title="Average Government Spending as a Share of GDP vs. Change in GDP per capita",
                log_x=log_x,
                log_y=log_y,
            )
            fig = apply_graph_stylings(fig)
            fig = add_country_lables(
                fig,
                df=all_subperiod_scatter_df,
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
                    data=all_subperiod_scatter_df.to_csv(
                        index=True, header=True
                    ).encode("utf-8"),
                    file_name="{0}_vs_{1}.csv".format(
                        x_title_no_brackets, y_title_no_brackets
                    ),
                    mime="text/csv",
                )
        elif selected_plot == "Heatmap":
            # st.dataframe(px.data.tips())
            if weight_pop:
                fig = px.density_heatmap(
                    all_subperiod_scatter_df,
                    x=x_title_no_brackets,
                    y=y_title_no_brackets,
                    z="Population",
                    histfunc="sum",
                    color_continuous_scale="Magma_r",
                )
            else:
                fig = px.density_heatmap(
                    all_subperiod_scatter_df,
                    x=x_title_no_brackets,
                    y=y_title_no_brackets,
                    color_continuous_scale="Magma_r",
                )
            # fig = apply_graph_stylings(fig)
            fig = add_country_lables(
                fig,
                df=all_subperiod_scatter_df,
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
                    data=all_subperiod_scatter_df.to_csv(
                        index=True, header=True
                    ).encode("utf-8"),
                    file_name="{0}_vs_{1}.csv".format(
                        x_title_no_brackets, y_title_no_brackets
                    ),
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
