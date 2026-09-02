from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.data import fetch_public_team
from fpl_decision_lab.modeling import METRIC_GLOSSARY
from fpl_decision_lab.optimizer import best_single_transfers, build_wildcard_squad, choose_starting_xi, plan_transfers
from fpl_decision_lab.paths import MODEL_JSON, PLAYER_FORECASTS_CSV, PLAYER_HISTORY_CSV, PLAYERS_CSV

st.set_page_config(page_title="FPL Streamlit Predictor", layout="wide")

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
.main { background: #f3f2f2; color: #201e1d; }
.block-container { max-width: 1500px; padding-top: 1.35rem; }
h1, h2, h3 { letter-spacing: 0; color: #201e1d; }
div[data-testid="stMetric"] { border-top: 2px solid #201e1d; padding-top: .45rem; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { border: 1px solid rgba(32,30,29,.16); border-radius: 2px; }
.pitch { background: #5d7f67; border: 2px solid #201e1d; min-height: 560px; padding: 18px; color: #f3f2f2; }
.pitch-row { display: flex; justify-content: center; gap: 12px; margin: 12px 0; flex-wrap: wrap; }
.player-card { width: 142px; min-height: 82px; background: rgba(243,242,242,.94); color: #201e1d; border-top: 4px solid #0088b0; padding: 7px 8px; box-shadow: 2px 2px 0 rgba(32,30,29,.22); }
.player-card strong { display: block; line-height: 1.05; font-size: 15px; }
.player-card span { display: block; font-size: 12px; color: rgba(32,30,29,.68); }
.badge { color: #d6006c; font-weight: 700; }
.bench { display: flex; gap: 10px; flex-wrap: wrap; border-top: 1px solid rgba(32,30,29,.25); padding-top: 12px; }
.metric-note { border-top: 2px solid #201e1d; padding-top: 8px; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

NUMERIC_COLUMNS = [
    "now_cost", "selected_by_percent", "total_points", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "expected_goals", "expected_assists", "expected_goal_involvements", "ict_index", "form",
    "points_per_game", "ep_next", "fixture_difficulty_next3", "fixture_difficulty_next5", "fixture_ease",
    "prior_found", "prior_team_found", "blended_points_per_90", "blended_xgi_per_90", "availability_factor",
    "predicted_next_gw", "predicted_next3", "predicted_next5", "baseline_next5", "optimizer_value",
    "transfer_score", "risk_score", "forecast_points", "gw", "horizon_index", "bank", "value",
]

DISPLAY_COLUMNS = [
    "web_name", "team", "position", "price", "selected_by_percent", "form", "points_per_game",
    "ep_next", "fixture_difficulty_next5", "predicted_next_gw", "predicted_next5", "transfer_score", "risk_score",
]

COLUMN_CONFIG = {
    "web_name": st.column_config.TextColumn("Player"),
    "team": st.column_config.TextColumn("Club"),
    "position": st.column_config.TextColumn("Pos"),
    "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
    "selected_by_percent": st.column_config.NumberColumn("Own %", format="%.1f%%"),
    "predicted_next_gw": st.column_config.NumberColumn("Next GW", help=METRIC_GLOSSARY["predicted_next_gw"], format="%.2f"),
    "predicted_next5": st.column_config.NumberColumn("Next 5", help=METRIC_GLOSSARY["predicted_next5"], format="%.2f"),
    "transfer_score": st.column_config.NumberColumn("Transfer score", help=METRIC_GLOSSARY["transfer_score"], format="%.2f"),
    "risk_score": st.column_config.NumberColumn("Risk", help=METRIC_GLOSSARY["risk_score"], format="%.2f"),
    "fixture_difficulty_next5": st.column_config.NumberColumn("FDR 5", help="Average official FPL fixture difficulty over the next five scheduled fixtures.", format="%.2f"),
}


@st.cache_data
def load_players() -> pd.DataFrame:
    if not PLAYERS_CSV.exists():
        st.error("Missing processed player data. Run scripts/refresh_all.py.")
        st.stop()
    data = pd.read_csv(PLAYERS_CSV)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["player_id"] = data["player_id"].astype(str)
    data["price"] = data["now_cost"] / 10
    return data


@st.cache_data
def load_history() -> pd.DataFrame:
    if not PLAYER_HISTORY_CSV.exists():
        return pd.DataFrame()
    data = pd.read_csv(PLAYER_HISTORY_CSV)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["player_id"] = data["player_id"].astype(str)
    return data


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    if not PLAYER_FORECASTS_CSV.exists():
        return pd.DataFrame()
    data = pd.read_csv(PLAYER_FORECASTS_CSV)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["player_id"] = data["player_id"].astype(str)
    return data


@st.cache_data
def load_model_summary() -> dict[str, Any]:
    if MODEL_JSON.exists():
        return json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    return {}


@st.cache_data(ttl=900)
def lookup_team(entry_id: int, event_id: int | None) -> dict[str, Any]:
    return fetch_public_team(entry_id, event_id)


def apply_global_filters(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data.copy()
    if st.session_state.position_filter != "All":
        filtered = filtered.loc[filtered["position"] == st.session_state.position_filter]
    if st.session_state.team_filter != "All":
        filtered = filtered.loc[filtered["team"] == st.session_state.team_filter]
    return filtered.loc[
        filtered["price"].between(st.session_state.price_range[0], st.session_state.price_range[1])
        & (filtered["minutes"] >= st.session_state.min_minutes)
        & (filtered["selected_by_percent"] <= st.session_state.max_ownership)
    ].copy()


def player_lookup(players: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {row["player_id"]: row.to_dict() for _, row in players.iterrows()}


def rows_for_squad(players: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return players.loc[players["player_id"].isin(ids)].copy()


def normalize_imported_team(payload: dict[str, Any], players: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    picks = payload.get("picks", {}).get("picks", [])
    pick_frame = pd.DataFrame(picks)
    if pick_frame.empty:
        return pd.DataFrame(), {}
    pick_frame["player_id"] = pick_frame["element"].astype(str)
    squad = players.merge(pick_frame[["player_id", "position", "multiplier", "is_captain", "is_vice_captain"]], on="player_id", how="inner", suffixes=("", "_pick"))
    squad["pick_position"] = squad["position_pick"]
    squad = squad.sort_values("pick_position")
    meta = payload.get("entry", {}) | payload.get("picks", {}).get("entry_history", {})
    return squad, meta


def card(player: dict[str, Any], score_key: str = "predicted_next_gw") -> str:
    captain = " <span class='badge'>C</span>" if player.get("is_captain") else ""
    vice = " <span class='badge'>V</span>" if player.get("is_vice_captain") else ""
    risk = float(player.get("risk_score", 0))
    border = "#d6006c" if risk >= 1.3 else "#edbb00" if risk >= .8 else "#0088b0"
    return (
        f"<div class='player-card' style='border-top-color:{border}'>"
        f"<strong>{player.get('web_name', '')}{captain}{vice}</strong>"
        f"<span>{player.get('team', '')} | {player.get('position', '')} | £{float(player.get('price', 0)):.1f}m</span>"
        f"<span>{float(player.get(score_key, 0)):.2f} pts | risk {float(player.get('risk_score', 0)):.2f}</span>"
        "</div>"
    )


def render_pitch(squad: pd.DataFrame, score_key: str = "predicted_next_gw") -> None:
    if squad.empty:
        st.info("Import a public FPL team or select a squad to see the pitch view.")
        return
    starters = squad.loc[squad.get("multiplier", 0) > 0].copy() if "multiplier" in squad else squad.head(11).copy()
    bench = squad.loc[squad.get("multiplier", 1) == 0].copy() if "multiplier" in squad else squad.iloc[11:].copy()
    html = ["<div class='pitch'>"]
    for position in ["FWD", "MID", "DEF", "GKP"]:
        row = starters.loc[starters["position"] == position].sort_values(score_key, ascending=False)
        if not row.empty:
            html.append("<div class='pitch-row'>" + "".join(card(item, score_key) for item in row.to_dict("records")) + "</div>")
    html.append("<div class='bench'>" + "".join(card(item, score_key) for item in bench.to_dict("records")) + "</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def metric_glossary() -> None:
    rows = [{"Metric": key, "Meaning": value} for key, value in METRIC_GLOSSARY.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_transfer_table(recommendations: list[dict[str, Any]]) -> None:
    rows = []
    for item in recommendations:
        rows.append({
            "Out": item["out"]["web_name"],
            "In": item["in"]["web_name"],
            "Pos": item["in"]["position"],
            "Club": item["in"]["team"],
            "Gain": item["gain"],
            "Bank after": item["net_budget"] / 10,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


players = load_players()
history = load_history()
forecasts = load_forecasts()
model_summary = load_model_summary()

st.markdown('<div style="height:6px;background:#201e1d"></div>', unsafe_allow_html=True)
left, right = st.columns([2.1, 1])
with left:
    st.caption("Fantasy Premier League | Streamlit | RandomForest | Transfer Planning")
    st.title("FPL Streamlit Predictor")
    st.write("Explore current FPL players, import a public team ID, inspect performance history, and plan transfers over the next five gameweeks.")
with right:
    st.metric("Players", f"{len(players):,}")
    st.metric("Current GW", f"{int(players['current_gameweek'].max())}")

with st.sidebar:
    st.header("Player filters")
    st.selectbox("Position", ["All", *sorted(players["position"].dropna().unique())], key="position_filter")
    st.selectbox("Club", ["All", *sorted(players["team"].dropna().unique())], key="team_filter")
    max_cost = float(players["price"].max())
    st.slider("Price range", 3.5, max(14.0, max_cost), (4.0, max(14.0, max_cost)), 0.1, key="price_range")
    st.slider("Minimum minutes", 0, int(players["minutes"].max()), 0, 90, key="min_minutes")
    st.slider("Maximum ownership %", 0.0, 100.0, 100.0, 0.5, key="max_ownership")

filtered = apply_global_filters(players)

home, my_team, player_lab, planner, data_notes = st.tabs(["Home", "My Team", "Player Lab", "Planner", "Data Notes"])

with home:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered players", f"{len(filtered):,}")
    c2.metric("Best next 5", f"{filtered['predicted_next5'].max():.1f}" if len(filtered) else "-")
    c3.metric("Model MAE", model_summary.get("modelDiagnostics", {}).get("randomForestMae", "-"))
    c4.metric("History rows", f"{len(history):,}")

    st.subheader("Transfer target landscape")
    chart_data = filtered.sort_values("predicted_next5", ascending=False).head(120)
    fig = px.scatter(
        chart_data,
        x="price",
        y="predicted_next5",
        size="selected_by_percent",
        color="position",
        hover_name="web_name",
        hover_data=["team", "form", "points_per_game", "fixture_difficulty_next5", "risk_score"],
        color_discrete_sequence=["#0088b0", "#d6006c", "#a07f00", "#201e1d"],
        labels={"price": "Price (£m)", "predicted_next5": "Projected next 5 GW points"},
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="#f3f2f2", plot_bgcolor="#f3f2f2")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(filtered.sort_values("transfer_score", ascending=False)[DISPLAY_COLUMNS], column_config=COLUMN_CONFIG, width="stretch", hide_index=True)

with my_team:
    st.subheader("Import a public FPL team")
    entry_col, override_col = st.columns([1, 1])
    with entry_col:
        entry_id = st.number_input("FPL team ID", min_value=1, step=1, value=2241815)
        event_id = st.number_input("Gameweek to import", min_value=1, max_value=38, value=int(players["current_gameweek"].max()), step=1)
    with override_col:
        manual_bank = st.number_input("Override bank (£m)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
        free_transfers = st.number_input("Free transfers", min_value=1, max_value=5, value=1, step=1)
    if st.button("Load team", type="primary"):
        try:
            payload = lookup_team(int(entry_id), int(event_id))
            squad, meta = normalize_imported_team(payload, players)
            st.session_state.imported_squad_ids = set(squad["player_id"].astype(str))
            st.session_state.imported_squad = squad
            st.session_state.imported_meta = meta
            st.session_state.manual_bank = manual_bank * 10
            st.session_state.free_transfers = int(free_transfers)
        except Exception as exc:
            st.error(f"Could not load that public team ID: {exc}")

    squad = st.session_state.get("imported_squad", pd.DataFrame())
    meta = st.session_state.get("imported_meta", {})
    if not squad.empty:
        st.write(f"**{meta.get('name', 'Imported team')}** | Overall points: {meta.get('summary_overall_points', meta.get('total_points', '-'))} | Bank: £{float(meta.get('bank', manual_bank * 10)) / 10:.1f}m")
        render_pitch(squad)
        recommendations = best_single_transfers(players.to_dict("records"), set(squad["player_id"].astype(str)), manual_bank * 10, int(free_transfers), limit=12, score_key="predicted_next5")
        st.subheader("Best immediate transfer options")
        render_transfer_table(recommendations)

with player_lab:
    st.subheader("Player performance history")
    selected = st.selectbox("Player", players.sort_values("web_name")["web_name"].tolist())
    player = players.loc[players["web_name"] == selected].sort_values("predicted_next5", ascending=False).iloc[0]
    player_history = history.loc[history["player_id"] == str(player["player_id"])] if not history.empty else pd.DataFrame()
    player_forecasts = forecasts.loc[forecasts["player_id"] == str(player["player_id"])] if not forecasts.empty else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"£{player['price']:.1f}m")
    c2.metric("Next GW", f"{player['predicted_next_gw']:.2f}")
    c3.metric("Next 5", f"{player['predicted_next5']:.2f}")
    c4.metric("Risk", f"{player['risk_score']:.2f}")
    if not player_history.empty:
        long = player_history.melt(id_vars=["gw"], value_vars=["total_points", "minutes", "expected_goal_involvements", "ict_index"], var_name="metric", value_name="metric_value")
        fig = px.line(long, x="gw", y="metric_value", color="metric", markers=True, color_discrete_sequence=["#0088b0", "#d6006c", "#a07f00", "#201e1d"])
        fig.update_layout(template="plotly_white", paper_bgcolor="#f3f2f2", plot_bgcolor="#f3f2f2")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(player_history.sort_values("gw"), width="stretch", hide_index=True)
    else:
        st.info("No checked current-season gameweek history is available for this player yet.")
    if not player_forecasts.empty:
        st.subheader("Upcoming forecast")
        st.dataframe(player_forecasts.sort_values("horizon_index"), width="stretch", hide_index=True)

with planner:
    st.subheader("Five-gameweek transfer planner")
    st.write("Use an imported public team from the My Team tab, or create a manual 15-player squad below.")
    default_ids = st.session_state.get("imported_squad_ids", set())
    options = players.sort_values("web_name")["web_name"].tolist()
    default_names = players.loc[players["player_id"].isin(default_ids), "web_name"].tolist()
    manual_names = st.multiselect("Squad", options=options, default=default_names)
    bank = st.number_input("Planner bank (£m)", min_value=0.0, max_value=50.0, value=float(st.session_state.get("manual_bank", 0.0)) / 10, step=0.1)
    free = st.number_input("Starting free transfers", min_value=1, max_value=5, value=int(st.session_state.get("free_transfers", 1)), step=1)
    if st.button("Build five-GW plan"):
        ids = set(players.loc[players["web_name"].isin(manual_names), "player_id"].astype(str))
        try:
            plans = plan_transfers(players.to_dict("records"), forecasts.to_dict("records"), ids, bank * 10, int(free), horizon=5)
            best = plans[0]
            st.metric("Best path projected score", f"{best['score']:.1f}")
            rows = []
            for step in best["steps"]:
                move_text = "Hold"
                if step["moves"]:
                    move_text = "; ".join(f"{move['out']['web_name']} -> {move['in']['web_name']}" for move in step["moves"])
                rows.append({"Week": step["week"], "Move": move_text, "Hits": step["hits"], "Captain": step["captain"]["web_name"], "Projected score": step["score"]})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(str(exc))

with data_notes:
    st.subheader("What the numbers mean")
    metric_glossary()
    st.subheader("Data coverage")
    coverage = model_summary.get("priorCoverage", {})
    st.write(
        f"The current snapshot has {len(players):,} players, {len(history):,} checked player-gameweek rows, "
        f"and {len(forecasts):,} player-fixture forecasts. Useful 2025-26 player priors are matched for "
        f"{coverage.get('playerPriorCoveragePct', 0)}% of current players."
    )
    st.write("Public team lookup uses the official public FPL entry endpoints. No login, password, or private session cookie is stored.")
