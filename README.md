# Tropang Foodie — Restaurant Popularity & Demand Explorer

An AI-class project that cleans a real scraped restaurant dataset, trains a
**classification model** (is this restaurant a "High" popularity performer?)
and a **regression model** (predicted annual demand score, 0–100), and serves
both through an interactive Streamlit app.

```
tropang-foodie-app/
├── app.py                     # Streamlit app (EDA + live predictions)
├── requirements.txt
├── data/
│   ├── Tropang_Foodie_Dataset_1.csv   # raw data
│   └── restaurants_clean.csv          # cleaned data (generated)
├── models/                    # trained model files (generated)
│   ├── classifier_pipeline.pkl
│   ├── regressor_pipeline.pkl
│   ├── ui_options.json
│   └── metrics.json
└── src/
    ├── data_prep.py            # Step 1: cleaning
    └── train_models.py         # Step 2: training
```

---

## Step 1 — Run it locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Clean the data
python src/data_prep.py

# 4. Train both models
python src/train_models.py

# 5. Launch the app
streamlit run app.py
```

Your browser should open `http://localhost:8501` with the app running.
Re-run steps 3–4 any time you swap in a new dataset — the app always loads
whatever is currently saved in `data/` and `models/`.

---

## Step 2 — Push the project to GitHub

```bash
# From inside the tropang-foodie-app folder
git init
git add .
git commit -m "Initial commit: data pipeline + Streamlit app"

# Create the repo on GitHub first (via github.com → New repository),
# then connect it:
git branch -M main
git remote add origin https://github.com/<your-username>/tropang-foodie-app.git
git push -u origin main
```

If you don't have Git installed or aren't sure you're authenticated, run
`git --version` and `gh auth status` (if using GitHub CLI) to check first.
For HTTPS pushes, GitHub will prompt for a **personal access token**
(not your password) — generate one under
GitHub → Settings → Developer settings → Personal access tokens.

**Note:** `models/*.pkl` are small here (~127-row dataset) so it's fine to
commit them directly. If your dataset grows large enough that the `.pkl`
files exceed a few MB, either add `models/` to `.gitignore` and have
Streamlit Cloud run `train_models.py` on startup, or use
[Git LFS](https://git-lfs.com/) for the binary files.

---

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Select:
   - Repository: `<your-username>/tropang-foodie-app`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy**. Streamlit Cloud will install everything in
   `requirements.txt` and boot the app automatically.
5. Your app gets a public URL like:
   `https://<your-username>-tropang-foodie-app.streamlit.app`

Any time you `git push` a change to `main`, Streamlit Cloud redeploys
automatically — no manual redeploy step needed.

---

## Design decisions worth knowing (for your class writeup)

- **Leakage guard:** `annual_demand_proxy_score_0_100` correlates strongly
  (0.67–0.81) with rating and review count, and `popularity_class` looks
  like it was thresholded directly from that same demand score. So the
  classifier never sees the demand score as an input, and the regressor
  never sees the popularity class — otherwise each model would effectively
  be "predicting" a value it was already handed.
- **Binary vs. 3-class target:** the raw `popularity_class` has only 3
  "Low" rows out of 127 — too few for any model to learn that pattern
  reliably. The classifier instead predicts **High vs. Not-High**, a
  reframing that's both trainable and still answers a useful business
  question.
- **Dropped feature:** `price_level_1_4` was ~81% missing, so it's kept in
  the cleaned CSV for reference but excluded from model inputs.
- **Small-data caveat:** the classifier scores look excellent (near-perfect)
  precisely because the dataset is small and the classes are cleanly
  separated by rating/reviews — that's a sign to gather more, messier data
  before trusting this for real decisions, not a sign the model is
  production-ready.
