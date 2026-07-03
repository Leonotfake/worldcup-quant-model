import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="World Cup Quant Model", layout="wide")

st.title("World Cup Quant Model")
st.write("App is working.")

teams = pd.DataFrame({
    "Team": ["France", "Argentina", "Spain", "England", "Brazil"],
    "Rating": [91.4, 89.0, 87.2, 85.5, 84.8],
    "Champion Probability": [0.30, 0.22, 0.15, 0.10, 0.08],
})

st.subheader("Current model output")
st.dataframe(teams)

st.bar_chart(teams.set_index("Team")["Champion Probability"])
