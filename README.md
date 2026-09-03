# FPL Streamlit Decision Lab

An interactive Fantasy Premier League portfolio project built to showcase Streamlit, football data visualisation, interpretable machine learning, and transfer optimization.

The repository is designed as a hybrid public portfolio:

- `docs/` is a static multi-page GitHub Pages site: the home page embeds the app, with separate explainer pages for features, metrics, and the RandomForest model.
- `app/streamlit_app.py` is the live Streamlit application, suitable for Streamlit Community Cloud.
- The current deployed app is expected at `https://fpl-app-predictor-sklxegssnh6exvzw2av6vg.streamlit.app/`.
- `data/processed/` contains committed sample outputs so the app can run without a live API call.
- `scripts/` contains the refresh pipeline for fetching current FPL data, rebuilding features, refreshing projections, and exporting static site data.

## What It Shows

- Current-season FPL data handling from public endpoints
- Player exploration by position, club, price, ownership, form, fixture difficulty, and projection
- Public FPL team lookup by entry ID with a pitch-style squad view
- Current-season player history charts and upcoming fixture forecasts
- RandomForest point forecasts with Ridge/form baselines and an accessible model explainer page
- Transfer recommendations and a five-gameweek transfer planner
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

Preview the GitHub Pages site:

```bash
python -m http.server 8000 --directory docs
```

## Data Refresh

The project ships with committed processed data so the app is self-contained. The current-season authority is the public FPL API for 2026-27. The refresh also fetches checked player gameweek histories, upcoming fixture forecasts, and Vaastav 2025-26 player/team/gameweek files as priors and training rows where available. Unmatched new players and new/promoted clubs receive neutral priors. To refresh from the live public FPL API:

```bash
python scripts/refresh_all.py
```

The GitHub Actions workflow in `.github/workflows/refresh-data.yml` can run this on a schedule and commit updated processed data back to the repository. This keeps the Streamlit app current without requiring private credentials.

## Repository Guide

```text
app/                 Streamlit user experience
data/processed/      Committed player, history, forecast, and model-summary outputs
data/raw/            Optional fetched API snapshots, ignored by git
docs/                GitHub Pages app home and explainer pages
scripts/             Pipeline entry points
src/fpl_decision_lab Reusable data, feature, model, and optimizer code
tests/               Contract and logic tests
```

## Streamlit Pages

The app includes:

- `Home`: fast player discovery and top transfer targets
- `My Team`: public FPL team-ID import, pitch view, and immediate transfer advice
- `Player Lab`: current-season gameweek history and future fixture forecasts
- `Planner`: five-gameweek transfer path planning
- `Model Explainer`: a separate, accessible explanation of the RandomForest model
- `Data Notes`: metric glossary and coverage notes

## Limits

This is a decision-support and portfolio project, not guaranteed FPL advice. Public FPL data does not capture injuries, tactical role changes, press conferences, likely minutes, private bookmaker prices, or human context. The model views should be read alongside the backtest and caveats in the app.


## GitHub Pages

The static site is split across `index.html`, `what-it-does.html`, `metric-glossary.html`, and `random-forest.html`. Set Pages `Source` to `GitHub Actions`. The workflow deploys `docs/` on every push to `main`; scheduled and manual runs also refresh data before deploying. The public Pages URL should become:

```text
https://elambden.github.io/fpl-streamlit-predictor/
```
