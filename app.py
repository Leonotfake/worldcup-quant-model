import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="World Cup Quant Model", layout="wide")

st.title("World Cup Quant Model")
st.write("Early MVP for World Cup team ratings and match prediction.")

teams = pd.DataFrame({
    "Team": ["France", "Argentina", "Spain", "England", "Brazil"],
    "Rating": [91.4, 89.0, 87.2, 85.5, 84.8],
    "Champion Probability": [0.30, 0.22, 0.15, 0.10, 0.08],
})

st.subheader("Current model output")
st.dataframe(teams, use_container_width=True)

st.subheader("Champion probability")
st.bar_chart(teams.set_index("Team")["Champion Probability"])

st.subheader("Match Predictor")

team_names = teams["Team"].tolist()

col1, col2 = st.columns(2)

with col1:
    team_a = st.selectbox("Team A", team_names, index=0)

with col2:
    team_b = st.selectbox("Team B", team_names, index=1)

rating_a = teams.loc[teams["Team"] == team_a, "Rating"].iloc[0]
rating_b = teams.loc[teams["Team"] == team_b, "Rating"].iloc[0]

rating_diff = rating_a - rating_b

prob_a = 1 / (1 + np.exp(-0.08 * rating_diff))
prob_b = 1 - prob_a

metric1, metric2 = st.columns(2)

metric1.metric(f"{team_a} win probability", f"{prob_a:.1%}")
metric2.metric(f"{team_b} win probability", f"{prob_b:.1%}")

st.write(f"Rating difference: {team_a} {rating_diff:+.1f} vs {team_b}")
