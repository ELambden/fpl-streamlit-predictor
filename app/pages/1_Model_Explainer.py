from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fpl_decision_lab.paths import MODEL_JSON

st.set_page_config(page_title="Model Explainer", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
    .block-container { max-width: 1180px; padding-top: 1.5rem; }
    div[data-testid="stMetric"] { border-top: 2px solid #201e1d; padding-top: .45rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

summary = json.loads(MODEL_JSON.read_text(encoding="utf-8")) if MODEL_JSON.exists() else {}
diag = summary.get("modelDiagnostics", {})
coverage = summary.get("priorCoverage", {})

st.markdown('<div style="height:6px;background:#201e1d"></div>', unsafe_allow_html=True)
st.caption("Machine learning primer")
st.title("How the predictor thinks")
st.write(
    "The main model is a RandomForestRegressor. In plain English, it builds lots of small decision trees, "
    "lets each tree make a points estimate, and averages them. That is useful for FPL because player value "
    "is not perfectly linear: minutes, role, fixtures, ownership, and recent output interact in awkward ways."
)

c1, c2, c3 = st.columns(3)
c1.metric("Training rows", f"{diag.get('trainingRows', 0):,}")
c2.metric("RandomForest MAE", diag.get("randomForestMae", "-"))
c3.metric("Ridge MAE", diag.get("ridgeMae", "-"))

st.subheader("Why RandomForest here?")
st.write(
    "A linear model is easy to explain, but it assumes every feature pushes points up or down in a straight line. "
    "FPL is messier than that. A defender with strong minutes and kind fixtures behaves differently from a rotation "
    "attacker with the same price. A RandomForest can learn those bends without asking us to hand-code every rule."
)
st.write(
    "The tradeoff is that it is less transparent than Ridge regression. So the app keeps Ridge and simple form-based "
    "baselines beside it, and exposes feature importance rather than pretending the model is magic."
)

importance = diag.get("featureImportance", {})
if importance:
    frame = pd.DataFrame({"Feature": list(importance.keys()), "Importance": list(importance.values())})
    fig = px.bar(frame, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#d6006c", "#0088b0"])
    fig.update_layout(template="plotly_white", paper_bgcolor="#f3f2f2", plot_bgcolor="#f3f2f2")
    st.plotly_chart(fig, width="stretch")

st.subheader("What feeds the model")
st.write(
    "The model trains on 2025-26 gameweek rows plus checked 2026-27 rows. It looks at recent points, minutes, starts, "
    "expected goal involvement, ICT, fixture difficulty, home/away, price, ownership, and lightweight team-strength priors. "
    "For the current season, 2026-27 data is always the authority. Prior-season player and team information is only a stabiliser."
)
st.dataframe(
    pd.DataFrame([
        {"Feature group": "Recent role", "Examples": "minutes, starts, recent points", "Why it helps": "Avoids overrating players who scored once but rarely play."},
        {"Feature group": "Attacking signal", "Examples": "xGI, ICT", "Why it helps": "Separates repeatable involvement from noisy goals and assists."},
        {"Feature group": "Fixture context", "Examples": "FDR, home/away", "Why it helps": "A good player against difficult opponents is not the same short-term bet."},
        {"Feature group": "Market context", "Examples": "price, ownership", "Why it helps": "Helps compare value and identify credible differentials."},
    ]),
    width="stretch",
    hide_index=True,
)

st.subheader("What it does not know")
st.write(
    "It does not read press conferences, injuries beyond public FPL status, tactical leaks, or your exact purchase prices. "
    "That is why the app calls the output advice, not certainty. The useful bit is narrowing the search and making tradeoffs visible."
)

st.subheader("Coverage")
st.write(
    f"Useful 2025-26 player priors matched for {coverage.get('playerPriorCoveragePct', 0)}% of current players. "
    f"Team priors matched for {coverage.get('teamPriorCoveragePct', 0)}% of player rows. New players and new/promoted clubs fall back to neutral priors."
)
