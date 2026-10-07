# Entry point (Streamlit Community Cloud runs this file): page config + navigation
import streamlit as st

st.set_page_config(
    page_title="Pitcher Valuation Dashboard",
    layout="wide",
)

home = st.Page("home.py", title="Home", default=True)
dashboard = st.Page("pitchers.py", title="Dashboard", url_path="dashboard")

st.navigation([home, dashboard]).run()
