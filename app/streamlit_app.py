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
    "minutes_per_game", "bonus_per_game", "bps_per_game", "defcons_per_90", "defcon_success_pct",
]

DISPLAY_COLUMNS = [
    "web_name", "team", "position", "price", "selected_by_percent", "form", "points_per_game",
    "expected_goals", "defcons_per_90", "defcon_success_pct", "bps_per_game", "predicted_next5", "transfer_score", "risk_score",
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
    "Defcons/90": "defcons_per_90",
    "Defcon Success %": "defcon_success_pct",
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
    "defcons_per_90": st.column_config.NumberColumn("Defcons/90", format="%.2f"),
    "defcon_success_pct": st.column_config.NumberColumn("Defcon Success %", format="%.1f%%"),
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
    for column in ["bonus_per_game", "bps_per_game", "defcons_per_90", "defcon_success_pct"]:
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
        & (filtered["defcons_per_90"] >= float(st.session_state.min_defcons_per90))
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


def render_decision_table(data: pd.DataFrame, row_limit: int) -> None:
    if data.empty:
        st.info("No players match the current filters.")
        return

    visible = data.head(row_limit).copy().reset_index(drop=True)
    table = visible[[
        "player_id", "player_label", "web_name", "team", "position", "price", "selected_by_percent",
        "form", "minutes_per_game", "expected_goals", "expected_assists", "expected_goal_involvements",
        "defcons_per_90", "defcon_success_pct", "bps_per_game", "predicted_next5",
        "transfer_score", "risk_score",
    ]].copy()

    event = st.dataframe(
        table,
        column_config={
            **COLUMN_CONFIG,
            "player_id": None,
            "player_label": None,
            "expected_assists": st.column_config.NumberColumn("xA", format="%.2f"),
            "expected_goal_involvements": st.column_config.NumberColumn("xGI", format="%.2f"),
            "minutes_per_game": st.column_config.NumberColumn("Min/G", format="%.0f"),
            "transfer_score": st.column_config.ProgressColumn(
                "Transfer score",
                help=METRIC_GLOSSARY["transfer_score"],
                min_value=0.0,
                max_value=max(float(table["transfer_score"].max()), 1.0),
                format="%.2f",
            ),
            "risk_score": st.column_config.ProgressColumn(
                "Risk",
                help=METRIC_GLOSSARY["risk_score"],
                min_value=0.0,
                max_value=max(float(table["risk_score"].max()), 1.0),
                format="%.2f",
            ),
        },
        column_order=[
            "web_name", "team", "position", "price", "selected_by_percent", "form", "minutes_per_game",
            "expected_goals", "expected_assists", "expected_goal_involvements", "defcons_per_90",
            "defcon_success_pct", "bps_per_game", "predicted_next5", "transfer_score", "risk_score",
        ],
        width="stretch",
        hide_index=True,
        key="decision_table",
        on_select="rerun",
        selection_mode="multi-row",
    )

    selected_rows = list(getattr(getattr(event, "selection", None), "rows", []))
    if selected_rows:
        selected = table.iloc[selected_rows]["player_label"].tolist()
        st.caption(f"Selected for removal: {', '.join(selected[:4])}" + ("..." if len(selected) > 4 else ""))
    if st.button("Remove selected players", disabled=not selected_rows):
        st.session_state.pending_remove_players = table.iloc[selected_rows]["player_label"].tolist()
        st.rerun()


def render_axis_chart(data: pd.DataFrame, x_label: str, y_label: str) -> None:
    if data.empty:
        st.info("No chart data matches the current filters.")
        return
    import plotly.express as px

    x_col = CHART_METRICS[x_label]
    y_col = CHART_METRICS[y_label]
    chart = pd.DataFrame({
        "Player": data["web_name"],
        "Club": data["team"],
        "Position": data["position"],
        "Price": data["price"],
        "Ownership": data["selected_by_percent"],
        "Form": data["form"],
        "Next 5": data["predicted_next5"],
        "Risk": data["risk_score"],
        "Defcons/90": data["defcons_per_90"],
        "Defcon Success %": data["defcon_success_pct"],
        "x_value": data[x_col],
        "y_value": data[y_col],
    }).dropna(subset=["x_value", "y_value"])
    fig = px.scatter(
        chart,
        x="x_value",
        y="y_value",
        color="Position",
        size="Ownership",
        hover_name="Player",
        hover_data={
            "Club": True,
            "Price": ":.1f",
            "Ownership": ":.1f",
            "Form": ":.1f",
            "Next 5": ":.2f",
            "Risk": ":.2f",
            "Defcons/90": ":.2f",
            "Defcon Success %": ":.1f",
            "x_value": False,
            "y_value": False,
        },
        labels={"x_value": x_label, "y_value": y_label, "Ownership": "Ownership %"},
        color_discrete_sequence=["#0088b0", "#d6006c", "#a07f00", "#201e1d"],
        height=430,
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="#f3f2f2", plot_bgcolor="#f3f2f2")
    st.plotly_chart(fig, width="stretch")


def render_home(players: pd.DataFrame, filtered: pd.DataFrame, model_summary: dict[str, Any]) -> None:
    st.caption("Fantasy Premier League | Decision Lab")
    st.title("FPL Streamlit Predictor")
    st.write("Compare players, tune the axes, remove noisy candidates, and plan transfers over the next five gameweeks.")


    metric_names = list(CHART_METRICS)
    axis_1, axis_2 = st.columns([1, 1])
    with axis_1:
        x_axis = st.selectbox("X axis", metric_names, index=metric_names.index("Price"))
    with axis_2:
        y_axis = st.selectbox("Y axis", metric_names, index=metric_names.index("Predicted next 5"))

    render_axis_chart(filtered, x_axis, y_axis)

    st.subheader("Decision table")
    table_controls = st.columns([1, 1, 1])
    with table_controls[0]:
        sort_by = st.selectbox("Rank table by", ["transfer_score", "predicted_next5", "defcons_per_90", "defcon_success_pct", "bps_per_game", "form", "price", "risk_score"], index=0)
    with table_controls[1]:
        row_limit = st.slider("Players shown", 10, 120, 40, 10)
    with table_controls[2]:
        ascending = st.toggle("Lowest first", value=sort_by == "risk_score")

    ranked = filtered.sort_values(sort_by, ascending=ascending)
    render_decision_table(ranked, row_limit)


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
    c4.metric("Defcons/90", f"{player['defcons_per_90']:.2f}")
    if not player_history.empty:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        hidden = {"player_id", "web_name", "team", "position", "fixture", "opponent_team_id", "was_home"}
        metric_columns = [
            column for column in player_history.columns
            if column not in hidden and column != "gw" and pd.api.types.is_numeric_dtype(player_history[column])
        ]
        label_by_column = {column: column.replace("_", " ").title() for column in metric_columns}
        label_by_column.update({
            "total_points": "Points",
            "expected_goals": "xG",
            "expected_assists": "xA",
            "expected_goal_involvements": "xGI",
            "ict_index": "ICT",
            "bps": "BPS",
            "defensive_contribution": "DefCons",
            "transfers_in": "Transfers in",
            "transfers_out": "Transfers out",
        })
        columns_by_label = {label_by_column[column]: column for column in metric_columns}
        labels = list(columns_by_label)
        primary_default = labels.index("Points") if "Points" in labels else 0
        secondary_default = labels.index("Minutes") if "Minutes" in labels else min(1, len(labels) - 1)
        y_primary = st.selectbox("Primary y-axis", labels, index=primary_default)
        y_secondary = st.selectbox("Secondary y-axis", ["None", *labels], index=secondary_default + 1 if labels else 0)

        ordered = player_history.sort_values("gw")
        fig = make_subplots(specs=[[{"secondary_y": y_secondary != "None"}]])
        primary_col = columns_by_label[y_primary]
        fig.add_trace(
            go.Scatter(x=ordered["gw"], y=ordered[primary_col], mode="lines+markers", name=y_primary, line={"color": "#0088b0", "width": 3}),
            secondary_y=False,
        )
        if y_secondary != "None":
            secondary_col = columns_by_label[y_secondary]
            fig.add_trace(
                go.Scatter(x=ordered["gw"], y=ordered[secondary_col], mode="lines+markers", name=y_secondary, line={"color": "#d6006c", "width": 3}),
                secondary_y=True,
            )
        fig.update_xaxes(title_text="Gameweek", dtick=1)
        fig.update_yaxes(title_text=y_primary, secondary_y=False)
        if y_secondary != "None":
            fig.update_yaxes(title_text=y_secondary, secondary_y=True)
        fig.update_layout(template="plotly_white", paper_bgcolor="#f3f2f2", plot_bgcolor="#f3f2f2", hovermode="x unified", height=430)
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

if "exclude_players" not in st.session_state:
    st.session_state.exclude_players = []
pending_removals = st.session_state.pop("pending_remove_players", [])
if pending_removals:
    merged_exclusions = list(dict.fromkeys(list(st.session_state.exclude_players) + list(pending_removals)))
    st.session_state.exclude_players = merged_exclusions
    st.session_state.exclude_players_select = merged_exclusions

with st.sidebar:
    st.header("Navigation")
    page = st.radio("View", ["Home", "My Team", "Player Lab", "Planner", "Data Notes"], label_visibility="collapsed")
    st.header("Player filters")
    st.selectbox("Position", ["All", *sorted(players["position"].dropna().unique())], key="position_filter")
    st.selectbox("Club", ["All", *team_options], key="team_filter")
    st.multiselect("Only include players", player_options, key="include_players")
    st.session_state.exclude_players = st.multiselect("Remove players", player_options, default=st.session_state.exclude_players, key="exclude_players_select")
    st.multiselect("Remove clubs", team_options, key="exclude_teams")
    max_cost = float(players["price"].max())
    st.slider("Price range", 3.5, max(14.0, max_cost), (4.0, max(14.0, max_cost)), 0.1, key="price_range")
    st.slider("Minimum total minutes", 0, int(players["minutes"].max()), 0, 90, key="min_minutes")
    st.slider("Minutes/game", 0, 90, 0, 5, key="min_minutes_per_game")
    st.select_slider("Minimum form", options=[0, 1, 2, 3, 4, 5, 6], value=0, key="min_form", format_func=lambda value: f"{value}+")
    st.select_slider("Minimum Defcons/90", options=list(range(13)), value=0, key="min_defcons_per90", format_func=lambda value: "12+" if value == 12 else str(value))
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
