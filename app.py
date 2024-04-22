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
    df = pd.read_csv(root_dir_path + "/data/forex_vs_gdp_per_capita.csv")
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


def add_country_lables(fig, df, countries, x_title, y_title):
    df_xy = df.loc[df["Country"].isin(countries), [x_title, y_title]]
    for i, (x, y) in enumerate(df_xy.itertuples(index=False)):
        fig.add_annotation(x=x, y=y, text=countries[i], showarrow=False)
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

    ### Side bar
    with st.sidebar:
        st.header("Universal options")
        with st.container(border=True):
            log_x = st.checkbox("log_x")
            log_y = st.checkbox("log_y")
        with st.container(border=True):
            show_pop = st.checkbox("Show Population")
        with st.container(border=True):
            display_countries = st.multiselect(
                label="Country Labels",
                options=sorted(forex_df["Country"].values),
            )

    ### Tabs
    tab_headers = {
        "tab1": "Salary vs. GDP per capita",
        "tab2": "Forex vs. GDP per capita",
    }
    tab1, tab2 = st.tabs([tab_headers[k] for k, v in tab_headers.items()])

    with tab1:
        ### Filters
        left_jobs_buffer, centre_jobs_col, right_jobs_buffer = st.columns([2, 8, 2])
        with centre_jobs_col:
            with st.container(border=True):
                selected_year = st.selectbox(
                    label="Year", options=list(range(1980, 2030)), index=44
                )
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
            .reset_index(drop=True)
        ).astype({"GDP_per_capita_USD": "float64"})
        # st.dataframe(job_df)

        ### Plot
        size = "Population" if show_pop else None
        x_title, y_title = "GDP_per_capita_USD", "Median_USD"
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
            title="Median pay of {0}s VS. GDP per capita ({1})".format(
                selected_job, selected_year
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
        )
        with st.container(border=True):
            st.plotly_chart(fig, theme=None, use_container_width=True)
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
        )
        with st.container(border=True):
            st.plotly_chart(fig, theme=None, use_container_width=True)


if __name__ == "__main__":
    main()
