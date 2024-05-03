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
        columns=["Unnamed: 0", "Code", "Continent"]
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
    # st.dataframe(spending_df)

    ### Side bar
    with st.sidebar:
        st.header("Plot Options")
        with st.container(border=True):
            log_x = st.checkbox("log_x")
            log_y = st.checkbox("log_y")
        with st.container(border=True):
            show_pop = st.checkbox("Population")
        with st.container(border=True):
            display_countries = st.multiselect(
                label="Country Labels",
                options=sorted(forex_df["Country"].values),
                placeholder="Add country labels to plots",
            )

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

        ### Apply filters
        job_df = (
            jobs_df.loc[
                (jobs_df["Job"] == selected_job) & (jobs_df["Year"] == selected_year),
                :,
            ]
            .drop(columns=["Unnamed: 0"])
            .drop_duplicates()
            .reset_index(drop=True)
        ).astype({"GDP_per_capita_USD": "float64"})

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
            # text=display_countries,
            trendline="ols",
            trendline_scope="overall",
            # trendline_options=dict(log_x=log_x, log_y=log_y),
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
            # text=display_countries,
            trendline="ols",
            trendline_scope="overall",
            # trendline_options=dict(log_x=log_x, log_y=log_y),
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
        ### Filters
        left_years_buffer, centre_years_col, right_years_buffer = st.columns([2, 8, 2])
        with centre_years_col:
            with st.container(border=True):
                spending_range = st.slider(
                    "Spending Range",
                    1986,
                    2011,
                    (1990, 2011),
                    help="The difference in government expenditure between two years.",
                )
                growth_range = st.slider(
                    "Growth Range",
                    1986,
                    2011,
                    (1990, 2011),
                    help="The difference in GDP per capita between two years.",
                )

        ### Apply filters
        spend_col = "Change in Government Expenditure as % of GDP ({0} - {1})".format(
            spending_range[0], spending_range[1]
        )
        growth_col = "Change in GDP per capita USD ({0} - {1})".format(
            growth_range[0], growth_range[1]
        )
        spending_df[spend_col] = spending_df.groupby(["Country"])[
            "Government Expenditure (IMF based on Mauro et al. (2015))"
        ].diff(spending_range[1] - spending_range[0])
        spending_df[growth_col] = spending_df.groupby(["Country"])[
            "GDP per capita, PPP (constant 2017 international $)"
        ].diff(growth_range[1] - growth_range[0])

        spending_df = spending_df.loc[
            spending_df["Year"].isin([spending_range[1], growth_range[1]]), :
        ]

        all_countries = spending_df["Country"].unique()
        for country in all_countries:
            spending_df.loc[spending_df["Country"] == country, spend_col] = (
                spending_df.loc[
                    (spending_df["Country"] == country)
                    & (spending_df["Year"] == spending_range[1]),
                    spend_col,
                ]
            ).values[0]
            spending_df.loc[spending_df["Country"] == country, growth_col] = (
                spending_df.loc[
                    (spending_df["Country"] == country)
                    & (spending_df["Year"] == growth_range[1]),
                    growth_col,
                ]
            ).values[0]
        # st.dataframe(spending_df)

        ### Plot
        size = "Population" if show_pop else None
        x_title, y_title = spend_col, growth_col
        filter_year = max(spending_range[1], growth_range[1])
        fig = px.scatter(
            spending_df.loc[spending_df["Year"] == filter_year],
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
            # text=display_countries,
            trendline="ols",
            trendline_scope="overall",
            # trendline_options=dict(log_x=log_x, log_y=log_y),
            trendline_color_override="black",
            title="Change in Government Spending as a Share of GDP vs. Change in GDP per capita",
            log_x=log_x,
            log_y=log_y,
        )
        fig = apply_graph_stylings(fig)
        fig = add_country_lables(
            fig,
            df=spending_df,
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
                data=spending_df.loc[
                    :, ["Country", "Region", "Population", x_title, y_title]
                ]
                .to_csv(index=True, header=True)
                .encode("utf-8"),
                file_name="{0}_vs_{1}.csv".format(x_title, y_title),
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
