# FPL Streamlit Decision Lab

An interactive Fantasy Premier League portfolio project built to showcase Streamlit, football data visualisation, interpretable machine learning, and transfer optimization.

The repository is designed as a hybrid public portfolio:

- `docs/` is a static GitHub Pages case-study wrapper styled to match the neighboring StatsBomb portfolio project.
- `app/streamlit_app.py` is the live Streamlit application, suitable for Streamlit Community Cloud.
- `data/processed/` contains committed sample outputs so the app can run without a live API call.
- `scripts/` contains the refresh pipeline for fetching current FPL data, rebuilding features, refreshing projections, and exporting static site data.

## What It Shows

- Current-season FPL data handling from public endpoints
- Player exploration by position, club, price, ownership, form, fixture difficulty, and projection
- Interpretable projection models with coefficient/model-audit views
- Transfer recommendations for generic managers and optional squad-specific inputs
- Wildcard-style squad optimization under FPL squad constraints
- A reproducible path from raw API data to deployed portfolio artifacts

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Preview the GitHub Pages wrapper:

```bash
python -m http.server 8000 --directory docs
```

## Data Refresh

The project ships with a compact sample dataset so the app is self-contained. To refresh from the live public FPL API:

```bash
python scripts/refresh_all.py
```

The GitHub Actions workflow in `.github/workflows/refresh-data.yml` can run this on a schedule and commit updated processed data back to the repository. This keeps the Streamlit app current without requiring private credentials.

## Repository Guide

```text
app/                 Streamlit user experience
data/processed/      Committed app-ready CSV and JSON outputs
data/raw/            Optional fetched API snapshots, ignored by git
docs/                GitHub Pages portfolio wrapper
scripts/             Pipeline entry points
src/fpl_decision_lab Reusable data, feature, model, and optimizer code
tests/               Contract and logic tests
```

## Limits

This is a decision-support and portfolio project, not guaranteed FPL advice. Public FPL data does not capture injuries, tactical role changes, press conferences, likely minutes, private bookmaker prices, or human context. The model views should be read alongside the backtest and caveats in the app.

