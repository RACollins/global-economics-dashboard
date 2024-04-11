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


##################
### App proper ###
##################


def main():
    st.title("Global Economics Dashboard")
    st.subheader("View various global economic data")

    ### Side bar
    with st.sidebar:
        tab_headers = {
            "tab1": "Salary vs. GDP per capita",
            "tab2": "Forex vs. GDP per capita",
        }
        st.header("Info...")
        with st.expander(tab_headers["tab1"]):
            st.write("Text1 " "Text2")
        with st.expander(tab_headers["tab2"]):
            st.write("Text1 " "Text2")

    ### Tabs
    tab1, tab2 = st.tabs([tab_headers[k] for k, v in tab_headers.items()])

    with tab1:
        ### Get dataframe
        df = pd.read_csv(root_dir_path + "/data/jobs_vs_gdp_per_capita.csv")

        ### Filters
        left_buffer_cols, left_col, right_col, right_buffer_cols = st.columns(
            [2, 4, 4, 2]
        )
        with left_col:
            with st.container(border=True):
                selected_year = st.selectbox(
                    label="Year", options=list(range(1980, 2030)), index=44
                )
                selected_job = st.selectbox(
                    label="Job",
                    options=["Bricklayer", "Doctor", "Nurse", "All"],
                    index=2,
                )
        with right_col:
            with st.container(border=True):
                log_x = st.checkbox("log_x")
                log_y = st.checkbox("log_y")

        ### Apply filters
        job_df = (
            df.loc[(df["Job"] == selected_job) & (df["Year"] == selected_year), :]
            .drop(columns=["Unnamed: 0"])
            .reset_index(drop=True)
        )
        # st.dataframe(job_df)

        ### Plot
        fig = px.scatter(
            job_df,
            x="GDP_per_capita_USD",
            y="Median_USD",
            color="Region",
            color_discrete_sequence=["red", "magenta", "goldenrod", "green", "blue"],
            category_orders={
                "Region": ["Asia", "Americas", "Africa", "Europe", "Oceania"]
            },
            size="Population",
            size_max=80,
            hover_data={"Country": True, "Population": True},
            # text="Country",
            trendline="ols",
            trendline_scope="overall",
            trendline_options=dict(log_x=False, log_y=False),
            trendline_color_override="black",
            title="Median pay of {0}s VS. GDP per capita ({1})".format(
                selected_job, selected_year
            ),
            log_x=log_x,
            log_y=log_y,
        )
        fig.update_traces(textposition="top center")
        with st.container(border=True):
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.info("Something should go here...")


if __name__ == "__main__":
    main()
