import json
import math

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="World Cup Quant Model", layout="wide")

BASE_GOAL_RATE = 1.35
MAX_GOALS = 6


@st.cache_data(ttl=3600)
def load_data():
    teams = pd.read_csv("data/teams.csv")
    matches = pd.read_csv("data/matches.csv")

    with open("data/data_version.json", "r", encoding="utf-8") as file:
        data_version = json.load(file)

    return teams, matches, data_version


def poisson_probability(goals, expected_goals):
    return math.exp(-expected_goals) * expected_goals**goals / math.factorial(goals)


def expected_goals(team_a, team_b, teams):
    row_a = teams.loc[teams["team"] == team_a].iloc[0]
    row_b = teams.loc[teams["team"] == team_b].iloc[0]

    avg_attack = teams["attack"].mean()
    avg_defense = teams["defense"].mean()

    attack_factor_a = row_a["attack"] / avg_attack
    attack_factor_b = row_b["attack"] / avg_attack

    defense_factor_a = avg_defense / row_a["defense"]
    defense_factor_b = avg_defense / row_b["defense"]

    rating_boost_a = 1 + ((row_a["rating"] - row_b["rating"]) / 100)
    rating_boost_b = 1 + ((row_b["rating"] - row_a["rating"]) / 100)

    lambda_a = BASE_GOAL_RATE * attack_factor_a * defense_factor_b * rating_boost_a
    lambda_b = BASE_GOAL_RATE * attack_factor_b * defense_factor_a * rating_boost_b

    lambda_a = float(np.clip(lambda_a, 0.25, 3.2))
    lambda_b = float(np.clip(lambda_b, 0.25, 3.2))

    return lambda_a, lambda_b


def score_matrix(lambda_a, lambda_b, max_goals=MAX_GOALS):
    rows = []

    for goals_a in range(max_goals + 1):
        for goals_b in range(max_goals + 1):
            prob = poisson_probability(goals_a, lambda_a) * poisson_probability(goals_b, lambda_b)
            rows.append({
                "team_a_goals": goals_a,
                "team_b_goals": goals_b,
                "probability": prob,
                "score": f"{goals_a}-{goals_b}",
            })

    return pd.DataFrame(rows)


def summarize_match(team_a, team_b, teams):
    lambda_a, lambda_b = expected_goals(team_a, team_b, teams)
    matrix = score_matrix(lambda_a, lambda_b)

    p_a_win = matrix.loc[matrix["team_a_goals"] > matrix["team_b_goals"], "probability"].sum()
    p_draw = matrix.loc[matrix["team_a_goals"] == matrix["team_b_goals"], "probability"].sum()
    p_b_win = matrix.loc[matrix["team_a_goals"] < matrix["team_b_goals"], "probability"].sum()

    most_likely = matrix.sort_values("probability", ascending=False).iloc[0]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "xg_a": lambda_a,
        "xg_b": lambda_b,
        "p_a_win": p_a_win,
        "p_draw": p_draw,
        "p_b_win": p_b_win,
        "most_likely_score": most_likely["score"],
        "most_likely_probability": most_likely["probability"],
        "matrix": matrix,
    }


def data_status_table(data_version):
    rows = []

    for name, info in data_version["data_sources"].items():
        rows.append({
            "data_table": name,
            "source": info["source"],
            "last_updated": info["last_updated"],
            "freshness": info["freshness"],
        })

    return pd.DataFrame(rows)


teams, matches, data_version = load_data()

st.title("World Cup Quant Model")
st.caption("Non-real-time, transparent football prediction model based on manually updated data.")

top1, top2, top3 = st.columns(3)
top1.metric("Model version", data_version["model_version"])
top2.metric("Last updated", data_version["last_updated"])
top3.metric("Data mode", "Manual CSV")

tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Match Predictor",
    "Schedule Prediction",
    "Data Status",
])

with tab1:
    st.subheader("Team Ratings")
    st.dataframe(teams.sort_values("rating", ascending=False), use_container_width=True)
    st.bar_chart(teams.set_index("team")["rating"])

with tab2:
    st.subheader("Single Match Score Predictor")

    team_names = teams["team"].tolist()

    col1, col2 = st.columns(2)

    with col1:
        team_a = st.selectbox("Team A", team_names, index=0)

    with col2:
        team_b = st.selectbox("Team B", team_names, index=1)

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        result = summarize_match(team_a, team_b, teams)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team_a} xG", f"{result['xg_a']:.2f}")
        c2.metric("Most likely score", result["most_likely_score"])
        c3.metric(f"{team_b} xG", f"{result['xg_b']:.2f}")

        c4, c5, c6 = st.columns(3)
        c4.metric(f"{team_a} win", f"{result['p_a_win']:.1%}")
        c5.metric("Draw", f"{result['p_draw']:.1%}")
        c6.metric(f"{team_b} win", f"{result['p_b_win']:.1%}")

        st.subheader("Most Likely Scores")
        top_scores = result["matrix"].sort_values("probability", ascending=False).head(10)
        top_scores = top_scores[["score", "probability"]].copy()
        top_scores["probability"] = top_scores["probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(top_scores, use_container_width=True)

        st.subheader("Score Probability Matrix")
        matrix_view = result["matrix"].pivot(
            index="team_a_goals",
            columns="team_b_goals",
            values="probability",
        )
        st.dataframe(matrix_view.style.format("{:.1%}"), use_container_width=True)

with tab3:
    st.subheader("Full Schedule Prediction")

    predictions = []

    for _, match in matches.iterrows():
        result = summarize_match(match["team_a"], match["team_b"], teams)
        predictions.append({
            "match_id": match["match_id"],
            "round": match["round"],
            "team_a": match["team_a"],
            "team_b": match["team_b"],
            "xg_a": round(result["xg_a"], 2),
            "xg_b": round(result["xg_b"], 2),
            "most_likely_score": result["most_likely_score"],
            "team_a_win": f"{result['p_a_win']:.1%}",
            "draw": f"{result['p_draw']:.1%}",
            "team_b_win": f"{result['p_b_win']:.1%}",
        })

    st.dataframe(pd.DataFrame(predictions), use_container_width=True)

with tab4:
    st.subheader("Data Status")
    st.write("This model is not real-time. Predictions are based on the latest manually updated data available in the repository.")
    st.dataframe(data_status_table(data_version), use_container_width=True)

    st.info("Next step: add team_stats.csv, match_results.csv, and historical_matches.csv for backtesting and machine learning.")
