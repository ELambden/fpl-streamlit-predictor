from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.optimizer import best_single_transfers, build_wildcard_squad, choose_starting_xi
from fpl_decision_lab.paths import MODEL_JSON, PLAYERS_CSV


st.set_page_config(page_title="FPL Streamlit Decision Lab", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
    .main { background: #f3f2f2; color: #201e1d; }
    .block-container { max-width: 1500px; padding-top: 1.5rem; }
    h1, h2, h3 { letter-spacing: 0; color: #201e1d; }
    div[data-testid="stMetric"] { border-top: 2px solid #201e1d; padding-top: .45rem; }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
    .stTabs [data-baseweb="tab"] { border: 1px solid rgba(32,30,29,.16); border-radius: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_players() -> pd.DataFrame:
    if not PLAYERS_CSV.exists():
        st.error("Missing processed player data. Run scripts/refresh_all.py or use the committed sample snapshot.")
        st.stop()
    data = pd.read_csv(PLAYERS_CSV)
    numeric = [
        "now_cost",
        "selected_by_percent",
        "total_points",
        "minutes",
        "starts",
        "goals_scored",
        "assists",
        "clean_sheets",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "ict_index",
        "form",
        "points_per_game",
        "fixture_difficulty_next3",
        "fixture_ease",
        "predicted_next_gw",
        "predicted_next3",
        "baseline_next3",
        "optimizer_value",
        "transfer_score",
        "risk_score",
    ]
    for column in numeric:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["price"] = data["now_cost"] / 10
    return data


@st.cache_data
def load_model_summary() -> dict:
    if MODEL_JSON.exists():
        return json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    return {}


players = load_players()
model_summary = load_model_summary()

st.markdown('<div style="height:6px;background:#201e1d"></div>', unsafe_allow_html=True)
left, right = st.columns([2, 1])
with left:
    st.caption("Fantasy Premier League | Streamlit | Machine Learning | Optimization")
    st.title("FPL Streamlit Decision Lab")
    st.write(
        "A live decision workbench for player discovery, fixture-adjusted projections, "
        "transfer planning, and constrained squad optimization."
    )
with right:
    st.metric("Players", f"{len(players):,}")
    st.metric("Mean 3-GW projection", f"{players['predicted_next3'].mean():.1f}")

with st.sidebar:
    st.header("Filters")
    positions = ["All", *sorted(players["position"].dropna().unique())]
    position = st.selectbox("Position", positions, index=0)
    teams = ["All", *sorted(players["team"].dropna().unique())]
    team = st.selectbox("Club", teams, index=0)
    max_cost = float(players["price"].max())
    price_range = st.slider("Price range", 3.5, max(14.0, max_cost), (4.0, max(14.0, max_cost)), 0.1)
    min_minutes = st.slider("Minimum minutes", 0, int(players["minutes"].max()), 0, 90)
    max_ownership = st.slider("Maximum ownership %", 0.0, 100.0, 100.0, 0.5)

filtered = players.copy()
if position != "All":
    filtered = filtered.loc[filtered["position"] == position]
if team != "All":
    filtered = filtered.loc[filtered["team"] == team]
filtered = filtered.loc[
    filtered["price"].between(price_range[0], price_range[1])
    & (filtered["minutes"] >= min_minutes)
    & (filtered["selected_by_percent"] <= max_ownership)
].copy()

tabs = st.tabs(["Explorer", "Transfers", "Optimizer", "Model Audit", "Data Notes"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered players", f"{len(filtered):,}")
    c2.metric("Best 3-GW projection", f"{filtered['predicted_next3'].max():.1f}" if len(filtered) else "-")
    c3.metric("Best value", f"{filtered['optimizer_value'].max():.2f}" if len(filtered) else "-")
    c4.metric("Lowest fixture difficulty", f"{filtered['fixture_difficulty_next3'].min():.1f}" if len(filtered) else "-")

    chart_data = filtered.sort_values("predicted_next3", ascending=False).head(80)
    fig = px.scatter(
        chart_data,
        x="price",
        y="predicted_next3",
        size="selected_by_percent",
        color="position",
        hover_name="web_name",
        hover_data=["team", "form", "points_per_game", "fixture_difficulty_next3"],
        color_discrete_sequence=["#0088b0", "#d6006c", "#a07f00", "#201e1d"],
        labels={"price": "Price (£m)", "predicted_next3": "Projected next 3 GW points"},
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="#f3f2f2", plot_bgcolor="#f3f2f2")
    st.plotly_chart(fig, width="stretch")

    show_cols = [
        "web_name",
        "team",
        "position",
        "price",
        "selected_by_percent",
        "form",
        "points_per_game",
        "fixture_difficulty_next3",
        "predicted_next_gw",
        "predicted_next3",
        "transfer_score",
    ]
    st.dataframe(
        filtered.sort_values("transfer_score", ascending=False)[show_cols],
        width="stretch",
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Transfer target board")
    st.write("Use the sidebar filters for generic target discovery, or add your current squad below for player-specific swaps.")
    generic = filtered.sort_values("transfer_score", ascending=False).head(20)
    st.dataframe(
        generic[["web_name", "team", "position", "price", "selected_by_percent", "predicted_next3", "transfer_score", "risk_score"]],
        width="stretch",
        hide_index=True,
    )

    with st.form("squad_form"):
        selected_names = st.multiselect(
            "Current squad",
            options=players.sort_values("web_name")["web_name"].tolist(),
            default=[],
        )
        bank = st.number_input("Bank available (£m)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        free_transfers = st.number_input("Free transfers", min_value=0, max_value=5, value=1, step=1)
        submitted = st.form_submit_button("Rank squad-specific transfers")
    if submitted and selected_names:
        squad_ids = set(players.loc[players["web_name"].isin(selected_names), "player_id"].astype(str))
        recommendations = best_single_transfers(players.to_dict("records"), squad_ids, bank * 10, int(free_transfers), limit=15)
        transfer_rows = [
            {
                "Out": item["out"]["web_name"],
                "In": item["in"]["web_name"],
                "Position": item["in"]["position"],
                "Club": item["in"]["team"],
                "Gain": item["gain"],
                "Bank after": item["net_budget"] / 10,
            }
            for item in recommendations
        ]
        st.dataframe(pd.DataFrame(transfer_rows), width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("Wildcard-style optimizer")
    budget = st.slider("Squad budget (£m)", 80.0, 110.0, 100.0, 0.5)
    pool = players.loc[players["minutes"] >= min_minutes].to_dict("records")
    try:
        squad = build_wildcard_squad(pool, budget * 10)
        xi = choose_starting_xi(squad)
        squad_frame = pd.DataFrame(squad)
        st.metric("Projected XI + captain", f"{xi['score_with_captain']:.1f}")
        st.write(f"Formation: {xi['formation']['DEF']}-{xi['formation']['MID']}-{xi['formation']['FWD']}")
        st.dataframe(
            squad_frame[["web_name", "team", "position", "price", "predicted_next_gw", "predicted_next3", "optimizer_value"]]
            .sort_values(["position", "predicted_next3"], ascending=[True, False]),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Captain: {xi['captain']['web_name']} | Vice captain: {xi['vice_captain']['web_name']}")
    except ValueError as exc:
        st.info(str(exc))

with tabs[3]:
    st.subheader("Model audit")
    st.json(model_summary)
    weights = model_summary.get("weights", {})
    if weights:
        weight_frame = pd.DataFrame({"Feature": list(weights), "Weight": list(weights.values())})
        fig = px.bar(weight_frame, x="Weight", y="Feature", orientation="h", color="Weight", color_continuous_scale=["#d6006c", "#0088b0"])
        fig.update_layout(template="plotly_white", paper_bgcolor="#f3f2f2", plot_bgcolor="#f3f2f2")
        st.plotly_chart(fig, width="stretch")
    st.write(
        "The committed sample uses an interpretable projection blend. After live refresh, "
        "this tab becomes the place to compare baselines, residuals, and feature influence."
    )

with tabs[4]:
    st.subheader("Data notes")
    st.write(
        "Data comes from the public Fantasy Premier League API. The repository keeps processed "
        "CSV/JSON snapshots committed so the public app remains self-contained."
    )
    st.write(
        "Model features use cumulative public fields and upcoming fixtures. The pipeline should "
        "avoid same-gameweek target leakage when historical gameweek data is added."
    )
    st.write(
        "Known limits: likely minutes, injuries, press conferences, tactical changes, price-change "
        "timing, and human risk appetite are not fully represented by public API data."
    )

