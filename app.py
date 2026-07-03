import json
import math

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="World Cup Quant Model", layout="wide")


@st.cache_data(ttl=3600)
def load_data():
    teams = pd.read_csv("data/teams.csv")
    team_stats = pd.read_csv("data/team_stats.csv")
    matches = pd.read_csv("data/matches.csv")

    with open("data/model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    with open("data/data_version.json", "r", encoding="utf-8") as f:
        data_version = json.load(f)

    return teams, team_stats, matches, config, data_version


def poisson_prob(goals, xg):
    return math.exp(-xg) * xg**goals / math.factorial(goals)


def build_model_table(teams, team_stats):
    data = teams.merge(team_stats, on="team", how="left")

    data["goals_per_match"] = data["goals"] / data["matches"]
    data["goals_against_per_match"] = data["goals_against"] / data["matches"]
    data["shots_on_target_per_match"] = data["shots_on_target"] / data["matches"]
    data["chances_created_per_match"] = data["chances_created"] / data["matches"]

    return data


def expected_goals(team_a, team_b, model_data, config):
    row_a = model_data.loc[model_data["team"] == team_a].iloc[0]
    row_b = model_data.loc[model_data["team"] == team_b].iloc[0]

    base = config["base_goal_rate"]
    limits = config["lambda_limits"]

    avg_attack = model_data["attack"].mean()
    avg_defense = model_data["defense"].mean()
    avg_elo = model_data["elo"].mean()
    avg_gpm = model_data["goals_per_match"].mean()
    avg_sot = model_data["shots_on_target_per_match"].mean()
    avg_chances = model_data["chances_created_per_match"].mean()

    attack_a = (
        0.35 * (row_a["attack"] / avg_attack)
        + 0.20 * (row_a["elo"] / avg_elo)
        + 0.20 * (row_a["goals_per_match"] / avg_gpm)
        + 0.15 * (row_a["shots_on_target_per_match"] / avg_sot)
        + 0.10 * (row_a["chances_created_per_match"] / avg_chances)
    )

    attack_b = (
        0.35 * (row_b["attack"] / avg_attack)
        + 0.20 * (row_b["elo"] / avg_elo)
        + 0.20 * (row_b["goals_per_match"] / avg_gpm)
        + 0.15 * (row_b["shots_on_target_per_match"] / avg_sot)
        + 0.10 * (row_b["chances_created_per_match"] / avg_chances)
    )

    defense_weakness_a = avg_defense / row_a["defense"]
    defense_weakness_b = avg_defense / row_b["defense"]

    rating_adj_a = 1 + ((row_a["rating"] - row_b["rating"]) / 120)
    rating_adj_b = 1 + ((row_b["rating"] - row_a["rating"]) / 120)

    xg_a = base * attack_a * defense_weakness_b * rating_adj_a
    xg_b = base * attack_b * defense_weakness_a * rating_adj_b

    xg_a = float(np.clip(xg_a, limits["min"], limits["max"]))
    xg_b = float(np.clip(xg_b, limits["min"], limits["max"]))

    return xg_a, xg_b


def score_matrix(xg_a, xg_b, max_goals):
    rows = []

    for a in range(max_goals + 1):
        for b in range(max_goals + 1):
            prob = poisson_prob(a, xg_a) * poisson_prob(b, xg_b)
            rows.append({
                "team_a_goals": a,
                "team_b_goals": b,
                "score": f"{a}-{b}",
                "probability": prob,
            })

    return pd.DataFrame(rows)


def summarize_match(team_a, team_b, model_data, config):
    xg_a, xg_b = expected_goals(team_a, team_b, model_data, config)
    matrix = score_matrix(xg_a, xg_b, config["max_goals"])

    p_a = matrix.loc[matrix["team_a_goals"] > matrix["team_b_goals"], "probability"].sum()
    p_d = matrix.loc[matrix["team_a_goals"] == matrix["team_b_goals"], "probability"].sum()
    p_b = matrix.loc[matrix["team_a_goals"] < matrix["team_b_goals"], "probability"].sum()

    top = matrix.sort_values("probability", ascending=False).iloc[0]

    return {
        "xg_a": xg_a,
        "xg_b": xg_b,
        "p_a": p_a,
        "p_d": p_d,
        "p_b": p_b,
        "most_likely_score": top["score"],
        "matrix": matrix,
    }


teams, team_stats, matches, config, data_version = load_data()
model_data = build_model_table(teams, team_stats)

st.title("World Cup Quant Model")
st.caption("Transparent, non-real-time football scoreline and schedule prediction model.")

c1, c2, c3 = st.columns(3)
c1.metric("Model version", data_version["model_version"])
c2.metric("Last updated", data_version["last_updated"])
c3.metric("Base goal rate", config["base_goal_rate"])

tabs = st.tabs(["Dashboard", "Match Predictor", "Score Matrix", "Schedule Prediction", "Data Status"])

with tabs[0]:
    st.subheader("Model Input Table")
    st.dataframe(model_data, use_container_width=True)

    st.subheader("Team Ratings")
    st.bar_chart(model_data.set_index("team")["rating"])

with tabs[1]:
    st.subheader("Match Predictor")

    team_names = model_data["team"].tolist()
    col1, col2 = st.columns(2)

    with col1:
        team_a = st.selectbox("Team A", team_names, index=0)

    with col2:
        team_b = st.selectbox("Team B", team_names, index=1)

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        result = summarize_match(team_a, team_b, model_data, config)

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} xG", f"{result['xg_a']:.2f}")
        m2.metric("Most likely score", result["most_likely_score"])
        m3.metric(f"{team_b} xG", f"{result['xg_b']:.2f}")

        p1, p2, p3 = st.columns(3)
        p1.metric(f"{team_a} win", f"{result['p_a']:.1%}")
        p2.metric("Draw", f"{result['p_d']:.1%}")
        p3.metric(f"{team_b} win", f"{result['p_b']:.1%}")

        st.subheader("Top Scorelines")
        top_scores = result["matrix"].sort_values("probability", ascending=False).head(10)
        top_scores = top_scores[["score", "probability"]].copy()
        top_scores["probability"] = top_scores["probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(top_scores, use_container_width=True)

with tabs[2]:
    st.subheader("Score Probability Matrix")

    team_names = model_data["team"].tolist()
    team_a = st.selectbox("Matrix Team A", team_names, index=0)
    team_b = st.selectbox("Matrix Team B", team_names, index=1)

    if team_a != team_b:
        result = summarize_match(team_a, team_b, model_data, config)
        matrix_view = result["matrix"].pivot(
            index="team_a_goals",
            columns="team_b_goals",
            values="probability",
        )
        st.dataframe(matrix_view.style.format("{:.1%}"), use_container_width=True)

with tabs[3]:
    st.subheader("Schedule Prediction")

    rows = []

    for _, match in matches.iterrows():
        result = summarize_match(match["team_a"], match["team_b"], model_data, config)
        rows.append({
            "match_id": match["match_id"],
            "round": match["round"],
            "team_a": match["team_a"],
            "team_b": match["team_b"],
            "xg_a": round(result["xg_a"], 2),
            "xg_b": round(result["xg_b"], 2),
            "most_likely_score": result["most_likely_score"],
            "team_a_win": f"{result['p_a']:.1%}",
            "draw": f"{result['p_d']:.1%}",
            "team_b_win": f"{result['p_b']:.1%}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tabs[4]:
    st.subheader("Data Status")
    st.write("This model is not real-time. Predictions use the latest manually updated data in the repository.")

    status_rows = []
    for name, info in data_version["data_sources"].items():
        status_rows.append({
            "data_table": name,
            "source": info["source"],
            "last_updated": info["last_updated"],
            "freshness": info["freshness"],
        })

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True)

    st.subheader("Current Model Config")
    st.json(config)
