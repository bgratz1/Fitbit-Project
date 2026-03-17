import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.artist import get

from part_4 import get_individual_summary, load_data

st.set_page_config(
    page_title = "title", 
    layout = "wide", 
    initial_sidebar_state = "expanded"
)

ID_LIST = [
    1503960366, 1624580081, 1644430081, 1844505072, 1927972279,
    2022484408, 2026352035, 2320127002, 2347167796, 2873212765,
    2891001357, 3372868164, 3977333714, 4020332650, 4057192912,
    4319703577, 4388161847, 4445114986, 4558609924, 4702921684,
    5553957443, 5577150313, 6117666160, 6290855005, 6391747486,
    6775888955, 6962181067, 7007744171, 7086361926, 8053475328,
    8253242879, 8378563200, 8583815059, 8792009665, 8877689391
]






st.title("Fitbit Dashboard")
st.header("Overview")


df = load_data()

# ID selector sidebar
with st.sidebar: 
    st.header("ID Selector")
    with st.form("user_selector"):
        selected_id = st.selectbox("Select user ID", ID_LIST)
        submitted = st.form_submit_button("Analyze User")

if submitted: 
    summary = get_individual_summary(df, selected_id)
    st.subheader(f"Summary for User {selected_id}")
    st.write(summary)




