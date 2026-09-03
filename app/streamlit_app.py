from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.data import fetch_public_team
from fpl_decision_lab.modeling import METRIC_GLOSSARY
from fpl_decision_lab.optimizer import best_single_transfers, plan_transfers
from fpl_decision_lab.paths import MODEL_JSON, PLAYER_FORECASTS_CSV, PLAYER_HISTORY_CSV, PLAYERS_CSV

st.set_page_config(page_title="FPL Streamlit Predictor", layout="wide")

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
.main { background: #f3f2f2; color: #201e1d; }
.block-container { max-width: 1500px; padding-top: 1rem; }
h1, h2, h3 { letter-spacing: 0; color: #201e1d; }
div[data-testid="stMetric"] { border-top: 2px solid #201e1d; padding-top: .45rem; }
.pitch { background: #5d7f67; border: 2px solid #201e1d; min-height: 560px; padding: 18px; color: #f3f2f2; }
.pitch-row { display: flex; justify-content: center; gap: 12px; margin: 12px 0; flex-wrap: wrap; }
.player-card { width: 142px; min-height: 82px; background: rgba(243,242,242,.94); color: #201e1d; border-top: 4px solid #0088b0; padding: 7px 8px; box-shadow: 2px 2px 0 rgba(32,30,29,.22); }
.player-card strong { display: block; line-height: 1.05; font-size: 15px; }
.player-card span { display: block; font-size: 12px; color: rgba(32,30,29,.68); }
.badge { color: #d6006c; font-weight: 700; }
.bench { display: flex; gap: 10px; flex-wrap: wrap; border-top: 1px solid rgba(32,30,29,.25); padding-top: 12px; }
.metric-note { border-top: 2px solid #201e1d; padding-top: 8px; }
.fast-note { color: rgba(32,30,29,.68); font-size: 13px; }
.decision-shell { border-top: 2px solid #201e1d; margin-top: 16px; overflow-x: auto; }
.decision-table { width: 100%; border-collapse: collapse; min-width: 1060px; background: rgba(255,255,255,.35); }
.decision-table th { color: rgba(32,30,29,.64); font-size: 11px; font-weight: 700; letter-spacing: .08em; padding: 9px 10px; text-align: left; text-transform: uppercase; white-space: nowrap; border-bottom: 1px solid rgba(32,30,29,.18); }
.decision-table td { padding: 10px; border-bottom: 1px solid rgba(32,30,29,.12); vertical-align: middle; }
.decision-table tr:hover { background: rgba(0,136,176,.07); }
.player-cell strong { display: block; font-size: 16px; line-height: 1.08; }
.player-cell span, .muted-cell { color: rgba(32,30,29,.62); font-size: 12px; }
.pos-pill { display: inline-flex; align-items: center; justify-content: center; min-width: 42px; border: 1px solid rgba(32,30,29,.22); border-radius: 99px; font-size: 11px; font-weight: 700; padding: 2px 8px; }
.score-bar { width: 100%; min-width: 96px; height: 7px; margin-top: 5px; background: rgba(32,30,29,.12); }
.score-bar span { display: block; height: 100%; background: #0088b0; }
.risk-low { color: #007f5f; font-weight: 700; }
.risk-mid { color: #a07f00; font-weight: 700; }
.risk-high { color: #d6006c; font-weight: 700; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

NUMERIC_COLUMNS = [
    "now_cost", "selected_by_percent", "total_points", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "bonus", "bps", "defensive_contribution", "defensive_contribution_per_90",
    "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded", "ict_index", "form",
    "points_per_game", "ep_next", "fixture_difficulty_next3", "fixture_difficulty_next5", "fixture_ease",
    "prior_found", "prior_team_found", "blended_points_per_90", "blended_xgi_per_90", "availability_factor",
    "predicted_next_gw", "predicted_next3", "predicted_next5", "baseline_next5", "optimizer_value",
    "transfer_score", "risk_score", "forecast_points", "gw", "horizon_index", "bank", "value",
    "minutes_per_game", "bonus_per_game", "bps_per_game", "defcons_per_90", "defcon_10_plus_pct",
]

DISPLAY_COLUMNS = [
    "web_name", "team", "position", "price", "selected_by_percent", "form", "points_per_game",
    "expected_goals", "defcons_per_90", "defcon_10_plus_pct", "bps_per_game", "predicted_next5", "transfer_score", "risk_score",
]

CHART_METRICS = {
    "Price": "price",
    "Predicted next GW": "predicted_next_gw",
    "Predicted next 5": "predicted_next5",
    "Transfer score": "transfer_score",
    "Risk score": "risk_score",
    "Ownership %": "selected_by_percent",
    "Form": "form",
    "Minutes/game": "minutes_per_game",
    "Points/game": "points_per_game",
    "xG": "expected_goals",
    "xA": "expected_assists",
    "xGI": "expected_goal_involvements",
    "DefCons/90": "defcons_per_90",
    "% 10+ DefCons": "defcon_10_plus_pct",
    "Bonus/game": "bonus_per_game",
    "BPS/game": "bps_per_game",
    "Fixture ease": "fixture_ease",
}

COLUMN_CONFIG = {
    "web_name": st.column_config.TextColumn("Player"),
    "team": st.column_config.TextColumn("Club"),
    "position": st.column_config.TextColumn("Pos"),
    "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
    "selected_by_percent": st.column_config.NumberColumn("Own %", format="%.1f%%"),
    "expected_goals": st.column_config.NumberColumn("xG", format="%.2f"),
    "defcons_per_90": st.column_config.NumberColumn("DefCons/90", format="%.2f"),
    "defcon_10_plus_pct": st.column_config.NumberColumn("10+ DefCon %", format="%.1f%%"),
    "bps_per_game": st.column_config.NumberColumn("BPS/G", format="%.2f"),
    "predicted_next_gw": st.column_config.NumberColumn("Next GW", help=METRIC_GLOSSARY["predicted_next_gw"], format="%.2f"),
    "predicted_next5": st.column_config.NumberColumn("Next 5", help=METRIC_GLOSSARY["predicted_next5"], format="%.2f"),
    "transfer_score": st.column_config.NumberColumn("Transfer score", help=METRIC_GLOSSARY["transfer_score"], format="%.2f"),
    "risk_score": st.column_config.NumberColumn("Risk", help=METRIC_GLOSSARY["risk_score"], format="%.2f"),
}


@st.cache_data(show_spinner=False)
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
    data["minutes_per_game"] = data.get("minutes_per_game", data["minutes"] / data["current_gameweek"].clip(lower=1)).fillna(0)
    for column in ["bonus_per_game", "bps_per_game", "defcons_per_90", "defcon_10_plus_pct"]:
        if column not in data:
            data[column] = 0.0
    data["player_label"] = data["web_name"] + " (" + data["team"] + ", " + data["position"] + ")"
    return data


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    if not PLAYER_HISTORY_CSV.exists():
        return pd.DataFrame()
    data = pd.read_csv(PLAYER_HISTORY_CSV)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["player_id"] = data["player_id"].astype(str)
    if "defensive_contribution" not in data:
        data["defensive_contribution"] = 0.0
    if "bonus" not in data:
        data["bonus"] = 0.0
    return data


@st.cache_data(show_spinner=False)
def load_forecasts() -> pd.DataFrame:
    if not PLAYER_FORECASTS_CSV.exists():
        return pd.DataFrame()
    data = pd.read_csv(PLAYER_FORECASTS_CSV)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["player_id"] = data["player_id"].astype(str)
    return data


@st.cache_data(show_spinner=False)
def load_model_summary() -> dict[str, Any]:
    if MODEL_JSON.exists():
        return json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    return {}


@st.cache_data(ttl=900, show_spinner=False)
def lookup_team(entry_id: int, event_id: int | None) -> dict[str, Any]:
    return fetch_public_team(entry_id, event_id)


def selected_ids(labels: list[str], players: pd.DataFrame) -> set[str]:
    if not labels:
        return set()
    return set(players.loc[players["player_label"].isin(labels), "player_id"].astype(str))


def apply_global_filters(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data
    include_ids = selected_ids(st.session_state.get("include_players", []), data)
    exclude_ids = selected_ids(st.session_state.get("exclude_players", []), data)
    if include_ids:
        filtered = filtered.loc[filtered["player_id"].isin(include_ids)]
    if exclude_ids:
        filtered = filtered.loc[~filtered["player_id"].isin(exclude_ids)]
    excluded_teams = st.session_state.get("exclude_teams", [])
    if excluded_teams:
        filtered = filtered.loc[~filtered["team"].isin(excluded_teams)]
    if st.session_state.position_filter != "All":
        filtered = filtered.loc[filtered["position"] == st.session_state.position_filter]
    if st.session_state.team_filter != "All":
        filtered = filtered.loc[filtered["team"] == st.session_state.team_filter]
    return filtered.loc[
        filtered["price"].between(st.session_state.price_range[0], st.session_state.price_range[1])
        & (filtered["minutes"] >= st.session_state.min_minutes)
        & (filtered["minutes_per_game"] >= st.session_state.min_minutes_per_game)
        & (filtered["selected_by_percent"] <= st.session_state.max_ownership)
        & (filtered["form"] >= st.session_state.min_form)
        & (filtered["defcons_per_90"].between(st.session_state.defcon_range[0], st.session_state.defcon_range[1]))
    ].copy()


def normalize_imported_team(payload: dict[str, Any], players: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    picks = payload.get("picks", {}).get("picks", [])
    pick_frame = pd.DataFrame(picks)
    if pick_frame.empty:
        return pd.DataFrame(), {}
    pick_frame["player_id"] = pick_frame["element"].astype(str)
    squad = players.merge(
        pick_frame[["player_id", "position", "multiplier", "is_captain", "is_vice_captain"]],
        on="player_id",
        how="inner",
        suffixes=("", "_pick"),
    )
    squad["pick_position"] = squad["position_pick"]
    squad = squad.sort_values("pick_position")
    meta = payload.get("entry", {}) | payload.get("picks", {}).get("entry_history", {})
    return squad, meta


def card(player: dict[str, Any], score_key: str = "predicted_next_gw") -> str:
    captain = " <span class='badge'>C</span>" if player.get("is_captain") else ""
    vice = " <span class='badge'>V</span>" if player.get("is_vice_captain") else ""
    risk = float(player.get("risk_score", 0))
    border = "#d6006c" if risk >= 1.3 else "#edbb00" if risk >= .8 else "#0088b0"
    name = html.escape(str(player.get("web_name", "")))
    team = html.escape(str(player.get("team", "")))
    position = html.escape(str(player.get("position", "")))
    return (
        f"<div class='player-card' style='border-top-color:{border}'>"
        f"<strong>{name}{captain}{vice}</strong>"
        f"<span>{team} | {position} | £{float(player.get('price', 0)):.1f}m</span>"
        f"<span>{float(player.get(score_key, 0)):.2f} pts | risk {float(player.get('risk_score', 0)):.2f}</span>"
        "</div>"
    )


def render_pitch(squad: pd.DataFrame, score_key: str = "predicted_next_gw") -> None:
    if squad.empty:
        st.info("Import a public FPL team or select a squad to see the pitch view.")
        return
    starters = squad.loc[squad.get("multiplier", 0) > 0].copy() if "multiplier" in squad else squad.head(11).copy()
    bench = squad.loc[squad.get("multiplier", 1) == 0].copy() if "multiplier" in squad else squad.iloc[11:].copy()
    markup = ["<div class='pitch'>"]
    for position in ["FWD", "MID", "DEF", "GKP"]:
        row = starters.loc[starters["position"] == position].sort_values(score_key, ascending=False)
        if not row.empty:
            markup.append("<div class='pitch-row'>" + "".join(card(item, score_key) for item in row.to_dict("records")) + "</div>")
    markup.append("<div class='bench'>" + "".join(card(item, score_key) for item in bench.to_dict("records")) + "</div>")
    markup.append("</div>")
    st.markdown("".join(markup), unsafe_allow_html=True)


def render_transfer_table(recommendations: list[dict[str, Any]]) -> None:
    if not recommendations:
        st.info("No legal upgrade found under the current budget and squad constraints.")
        return
    rows = [
        {
            "Out": item["out"]["web_name"],
            "In": item["in"]["web_name"],
            "Pos": item["in"]["position"],
            "Club": item["in"]["team"],
            "Gain": item["gain"],
            "Bank after": item["net_budget"] / 10,
        }
        for item in recommendations
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def risk_class(value: float) -> str:
    if value >= 1.3:
        return "risk-high"
    if value >= 0.8:
        return "risk-mid"
    return "risk-low"


def render_decision_table(data: pd.DataFrame, row_limit: int) -> None:
    if data.empty:
        st.info("No players match the current filters.")
        return
    visible = data.head(row_limit).copy()
    max_score = max(float(visible["transfer_score"].max()), 0.1)
    rows = []
    for player in visible.to_dict("records"):
        score = float(player.get("transfer_score", 0))
        bar_width = max(2, min(100, int(100 * max(score, 0) / max_score)))
        risk = float(player.get("risk_score", 0))
        rows.append(
            "<tr>"
            f"<td class='player-cell'><strong>{html.escape(str(player.get('web_name', '')))}</strong><span>{html.escape(str(player.get('full_name', '')))}</span></td>"
            f"<td>{html.escape(str(player.get('team', '')))}<br><span class='muted-cell'>{html.escape(str(player.get('status', '')))}</span></td>"
            f"<td><span class='pos-pill'>{html.escape(str(player.get('position', '')))}</span></td>"
            f"<td>£{float(player.get('price', 0)):.1f}m</td>"
            f"<td>{float(player.get('form', 0)):.1f}</td>"
            f"<td>{float(player.get('minutes_per_game', 0)):.0f}</td>"
            f"<td>{float(player.get('expected_goals', 0)):.2f}</td>"
            f"<td>{float(player.get('defcons_per_90', 0)):.2f}<br><span class='muted-cell'>{float(player.get('defcon_10_plus_pct', 0)):.0f}% 10+</span></td>"
            f"<td>{float(player.get('bps_per_game', 0)):.1f}</td>"
            f"<td>{float(player.get('predicted_next5', 0)):.2f}</td>"
            f"<td><strong>{score:.2f}</strong><div class='score-bar'><span style='width:{bar_width}%'></span></div></td>"
            f"<td class='{risk_class(risk)}'>{risk:.2f}</td>"
            "</tr>"
        )
    table = (
        "<div class='decision-shell'><table class='decision-table'>"
        "<thead><tr><th>Player</th><th>Club</th><th>Pos</th><th>Price</th><th>Form</th><th>Min/G</th><th>xG</th><th>DefCons</th><th>BPS/G</th><th>Next 5</th><th>Transfer</th><th>Risk</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)


def render_axis_chart(data: pd.DataFrame, x_label: str, y_label: str) -> None:
    if data.empty:
        st.info("No chart data matches the current filters.")
        return
    x_col = CHART_METRICS[x_label]
    y_col = CHART_METRICS[y_label]
    chart = data[["web_name", "team", "position", x_col, y_col, "selected_by_percent"]].dropna().copy()
    chart = chart.rename(columns={x_col: x_label, y_col: y_label, "selected_by_percent": "Ownership %"})
    st.scatter_chart(chart, x=x_label, y=y_label, color="position", size="Ownership %", height=420)


def render_home(players: pd.DataFrame, filtered: pd.DataFrame, model_summary: dict[str, Any]) -> None:
    st.caption("Fantasy Premier League | Decision Lab")
    st.title("FPL Streamlit Predictor")
    st.write("Compare players, tune the axes, remove noisy candidates, and plan transfers over the next five gameweeks.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered players", f"{len(filtered):,}")
    c2.metric("Best next 5", f"{filtered['predicted_next5'].max():.1f}" if len(filtered) else "-")
    c3.metric("Best DefCons/90", f"{filtered['defcons_per_90'].max():.1f}" if len(filtered) else "-")
    c4.metric("Model MAE", model_summary.get("modelDiagnostics", {}).get("randomForestMae", "-"))

    control_1, control_2, control_3 = st.columns([1, 1, 1])
    metric_names = list(CHART_METRICS)
    with control_1:
        x_axis = st.selectbox("X axis", metric_names, index=metric_names.index("Price"))
    with control_2:
        y_axis = st.selectbox("Y axis", metric_names, index=metric_names.index("Predicted next 5"))
    with control_3:
        sort_by = st.selectbox("Rank table by", ["transfer_score", "predicted_next5", "defcons_per_90", "defcon_10_plus_pct", "bps_per_game", "form", "price", "risk_score"], index=0)

    render_axis_chart(filtered, x_axis, y_axis)

    table_controls = st.columns([1, 1, 1])
    with table_controls[0]:
        row_limit = st.slider("Rows shown", 10, 120, 40, 10)
    with table_controls[1]:
        ascending = st.toggle("Lowest first", value=sort_by == "risk_score")
    with table_controls[2]:
        show_raw = st.toggle("Show compact grid", value=False)

    ranked = filtered.sort_values(sort_by, ascending=ascending)
    st.subheader("Decision table")
    render_decision_table(ranked, row_limit)
    if show_raw:
        st.dataframe(ranked.head(row_limit)[DISPLAY_COLUMNS], column_config=COLUMN_CONFIG, width="stretch", hide_index=True)


def render_my_team(players: pd.DataFrame) -> None:
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
        bank_value = float(meta.get("bank", manual_bank * 10)) / 10
        st.write(f"**{meta.get('name', 'Imported team')}** | Overall points: {meta.get('summary_overall_points', meta.get('total_points', '-'))} | Bank: £{bank_value:.1f}m")
        render_pitch(squad)
        recommendations = best_single_transfers(
            players.to_dict("records"),
            set(squad["player_id"].astype(str)),
            manual_bank * 10,
            int(free_transfers),
            limit=12,
            score_key="predicted_next5",
        )
        st.subheader("Best immediate transfer options")
        render_transfer_table(recommendations)


def render_player_lab(players: pd.DataFrame) -> None:
    st.subheader("Player performance history")
    selected = st.selectbox("Player", players.sort_values("web_name")["player_label"].tolist())
    player = players.loc[players["player_label"] == selected].iloc[0]
    history = load_history()
    forecasts = load_forecasts()
    player_history = history.loc[history["player_id"] == str(player["player_id"])] if not history.empty else pd.DataFrame()
    player_forecasts = forecasts.loc[forecasts["player_id"] == str(player["player_id"])] if not forecasts.empty else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"£{player['price']:.1f}m")
    c2.metric("Next GW", f"{player['predicted_next_gw']:.2f}")
    c3.metric("Next 5", f"{player['predicted_next5']:.2f}")
    c4.metric("DefCons/90", f"{player['defcons_per_90']:.2f}")
    if not player_history.empty:
        import plotly.express as px

        history_metrics = {
            "Points": "total_points",
            "Minutes": "minutes",
            "xG": "expected_goals",
            "xA": "expected_assists",
            "xGI": "expected_goal_involvements",
            "ICT": "ict_index",
            "BPS": "bps",
            "Bonus": "bonus",
            "DefCons": "defensive_contribution",
            "Transfers in": "transfers_in",
            "Transfers out": "transfers_out",
        }
        x_metric = st.selectbox("History X axis", list(history_metrics), index=list(history_metrics).index("Points"))
        y_metric = st.selectbox("History Y axis", list(history_metrics), index=list(history_metrics).index("Minutes"))
        fig = px.scatter(
            player_history,
            x=history_metrics[x_metric],
            y=history_metrics[y_metric],
            size="minutes",
            color="total_points",
            hover_data=["gw", "total_points", "minutes", "defensive_contribution", "bps", "bonus"],
            color_continuous_scale=["#d6006c", "#edbb00", "#0088b0"],
        )
        fig.update_layout(template="plotly_white", paper_bgcolor="#f3f2f2", plot_bgcolor="#f3f2f2")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(player_history.sort_values("gw"), width="stretch", hide_index=True)
    else:
        st.info("No checked current-season gameweek history is available for this player yet.")
    if not player_forecasts.empty:
        st.subheader("Upcoming forecast")
        st.dataframe(player_forecasts.sort_values("horizon_index"), width="stretch", hide_index=True)


def render_planner(players: pd.DataFrame) -> None:
    st.subheader("Five-gameweek transfer planner")
    st.write("Use an imported public team from My Team, or create a manual 15-player squad below.")
    default_ids = st.session_state.get("imported_squad_ids", set())
    options = players.sort_values("web_name")["player_label"].tolist()
    default_names = players.loc[players["player_id"].isin(default_ids), "player_label"].tolist()
    manual_names = st.multiselect("Squad", options=options, default=default_names)
    bank = st.number_input("Planner bank (£m)", min_value=0.0, max_value=50.0, value=float(st.session_state.get("manual_bank", 0.0)) / 10, step=0.1)
    free = st.number_input("Starting free transfers", min_value=1, max_value=5, value=int(st.session_state.get("free_transfers", 1)), step=1)
    if st.button("Build five-GW plan"):
        ids = set(players.loc[players["player_label"].isin(manual_names), "player_id"].astype(str))
        try:
            forecasts = load_forecasts()
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


def render_data_notes(players: pd.DataFrame, model_summary: dict[str, Any]) -> None:
    history = load_history()
    forecasts = load_forecasts()
    st.subheader("What the numbers mean")
    st.dataframe(pd.DataFrame([{"Metric": key, "Meaning": value} for key, value in METRIC_GLOSSARY.items()]), width="stretch", hide_index=True)
    st.subheader("Data coverage")
    coverage = model_summary.get("priorCoverage", {})
    st.write(
        f"The current snapshot has {len(players):,} players, {len(history):,} checked player-gameweek rows, "
        f"and {len(forecasts):,} player-fixture forecasts. Useful 2025-26 player priors are matched for "
        f"{coverage.get('playerPriorCoveragePct', 0)}% of current players."
    )
    st.write("Public team lookup uses the official public FPL entry endpoints. No login, password, or private session cookie is stored.")


players = load_players()
model_summary = load_model_summary()
player_options = players.sort_values("web_name")["player_label"].tolist()
team_options = sorted(players["team"].dropna().unique())

with st.sidebar:
    st.header("Navigation")
    page = st.radio("View", ["Home", "My Team", "Player Lab", "Planner", "Data Notes"], label_visibility="collapsed")
    st.header("Player filters")
    st.selectbox("Position", ["All", *sorted(players["position"].dropna().unique())], key="position_filter")
    st.selectbox("Club", ["All", *team_options], key="team_filter")
    st.multiselect("Only include players", player_options, key="include_players")
    st.multiselect("Remove players", player_options, key="exclude_players")
    st.multiselect("Remove clubs", team_options, key="exclude_teams")
    max_cost = float(players["price"].max())
    st.slider("Price range", 3.5, max(14.0, max_cost), (4.0, max(14.0, max_cost)), 0.1, key="price_range")
    st.slider("Minimum total minutes", 0, int(players["minutes"].max()), 0, 90, key="min_minutes")
    st.slider("Minutes/game", 0, 90, 0, 5, key="min_minutes_per_game")
    st.select_slider("Minimum form", options=[0, 1, 2, 3, 4, 5, 6], value=0, key="min_form", format_func=lambda value: f"{value}+")
    st.slider("DefCons/90", 0.0, max(10.0, float(players["defcons_per_90"].max())), (0.0, max(10.0, float(players["defcons_per_90"].max()))), 0.1, key="defcon_range")
    st.slider("Maximum ownership %", 0.0, 100.0, 100.0, 0.5, key="max_ownership")

filtered = apply_global_filters(players)
st.markdown('<div style="height:6px;background:#201e1d"></div>', unsafe_allow_html=True)

if page == "Home":
    render_home(players, filtered, model_summary)
elif page == "My Team":
    render_my_team(players)
elif page == "Player Lab":
    render_player_lab(players)
elif page == "Planner":
    render_planner(players)
else:
    render_data_notes(players, model_summary)
