# ATLAS Console + ML Lab (static web app)

The original ATLAS mission-control console (Team Binary Knights, NASA Space
Apps 2026) with the Lab 14 machine learning layer added as a new **ML Lab**
view. Every original view (Overview, Bases, Finder, Compare, Mars, Method)
works exactly as before. This folder is a fully self-contained static site:
no backend and no build step are needed.

## What is inside

- `index.html`, `app.js`, `styles.css`, `vendor/`: the original console,
  with three small additions in `app.js` (an "ML Lab" phase label, route and
  lazy-init hook) and one new rail item plus view in `index.html`.
- `ml-lab.js`: the ML Lab predict playground. The project's best trained
  regressor (StandardScaler + Linear Regression, CV R2 0.9817) is exported
  coefficient for coefficient and runs entirely in the browser.
- `static-data.js` + `static-api.js`: the console originally read a FastAPI
  backend at `/api/*`. This build bundles a snapshot of that backend's own
  responses and answers the same requests in the browser. Balanced rankings
  return the stored official scores untouched; custom weights rescore with
  the same formula the backend used. The CSV download buttons are served
  from `api/downloads/`.
- `outputs/`: the LOLA hillshade images the console displays, plus the four
  Lab 14 result plots under `outputs/ml/`.

## Deploy on Vercel (from the atlas-ml-cca repo)

1. Drop this `webapp/` folder into the root of the `atlas-ml-cca` repo and
   push.
2. In Vercel, import the `maryamsohail32/atlas-ml-cca` repository.
3. Set **Root Directory** to `webapp`.
4. Framework preset: **Other**. No build command, no output directory
   setting (the folder itself is the site).
5. Deploy. The Streamlit app elsewhere in the repo is untouched and keeps
   its own separate deployment.

## Run locally

Any static file server works, for example from inside this folder:

```
python -m http.server 8000
```

Then open http://localhost:8000 and use the **ML Lab** item in the left
rail.
