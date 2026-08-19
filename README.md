# VeriFact Pro — AI/ML Fake News Detection System
🔗 **Live Demo:** [https://fake-news-detection-verifact.onrender.com](https://fake-news-detection-verifact.onrender.com)

**Submitted by:** Ankita Kumari
**Project:** Final Year Project — AI/ML (Fake News Detection System)
**Date:** August 2026

---

[![Tests](https://github.com/Itsmeankita/fake-news-detection-verifact/actions/workflows/tests.yml/badge.svg)](https://github.com/Itsmeankita/fake-news-detection-verifact/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A full-featured fake news detection platform: an NLP model (TF-IDF + classic
ML, trained locally on CPU) wrapped in a multi-page Flask website with user
accounts, history, URL/batch detection, explainability, PDF reports, a media
literacy quiz, a browser extension, and an optional deep learning (DistilBERT)
comparison model.

Built as a final-year project. Core system runs entirely locally in VS Code.

---

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────────────┐
│   Browser   │ ───► │  Flask app   │ ───► │  TF-IDF Vectorizer  │
│ (or Chrome  │ ◄─── │   (app.py)   │ ◄─── │  + Logistic Reg.    │
│  extension) │      │              │      │  (model/model.pkl)  │
└─────────────┘      └──────┬───────┘      └────────────────────┘
                             │
                 ┌───────────┼────────────┐
                 ▼           ▼            ▼
           SQLite DB   Fact Check API   PDF/CSV
          (users, history)  (optional)   export
```

---

## Feature list

**Core detection**
1. Paste-text detection with REAL/FAKE verdict + confidence score
2. URL detection — paste a link, the article text is scraped and analyzed automatically
3. Batch detection — upload a CSV of articles, get verdicts for all of them at once
4. Compare mode — analyze two articles side by side
5. Explainability — see the specific words that pushed the verdict toward REAL or FAKE
6. Voice input — dictate article text via your microphone (browser speech-to-text)
7. Sample buttons — try a real-style or fake-style example instantly
8. PDF report export — download any result as a formatted PDF

**Accounts & history**
9. Sign up / log in (secure password hashing)
10. Personal dashboard — see your full detection history
11. Per-user stats (REAL vs FAKE counts, total checks)
12. Auto-generated API key per user
13. Admin panel — view all registered users and system-wide stats
14. Simple role system (admin vs regular user)

**Analytics & transparency**
15. Live accuracy / precision / recall / F1-score dashboard
16. Confusion matrix visualization
17. Model comparison chart (Logistic Regression vs Naive Bayes vs Passive Aggressive)
18. Dataset class balance chart
19. Word clouds — most common words in real vs fake training articles
20. Full dataset & preprocessing stats table
21. "How it works" page — full pipeline explained step by step

**Engagement & education**
22. "Spot the fake news" interactive quiz with instant feedback and scoring
23. FAQ page addressing model limitations honestly
24. Dark / light theme toggle (persisted across visits)

**Engineering**
25. REST API (`/api/predict`, `/api/predict-url`, `/api/predict-batch`, `/api/compare`, `/api/export-pdf`, `/api/health`, `/api/model-info`)
26. SQLite database (via SQLAlchemy) for users & history — zero external setup
27. Modular codebase — shared prediction logic, shared preprocessing, reusable explainability module
28. Graceful degradation — every page works even before the model is trained, with a clear message instead of crashing
29. `.gitignore` configured for a clean GitHub push
30. Deployment-ready: `Procfile` + `runtime.txt` for Render/Railway, plus a full `Dockerfile` + `docker-compose.yml`
31. API rate limiting (Flask-Limiter — 30 predictions/minute per IP, 200 requests/hour default)
32. Rotating file logging (`logs/app.log`) — every prediction is logged with verdict + confidence
33. `/api/health` liveness endpoint for uptime monitors / deployment platforms
34. `/api/model-info` endpoint — machine-readable API documentation
35. Global model explainability — Analytics page shows the strongest FAKE/REAL-indicating words across the *entire* vocabulary, not just per-request
36. Confidence calibration (reliability) diagram — checks if confidence scores are trustworthy, not just accuracy
37. Config via `.env` file (`python-dotenv`) — secrets never hardcoded
38. Unit test suite (`pytest`, 18 tests) covering pages, auth, and the prediction API
39. GitHub Actions CI — tests run automatically on every push
40. Companion Chrome extension (`browser-extension/`) — right-click any selected text on any webpage → instant verdict
41. Optional DistilBERT deep-learning comparison model (`model/train_bert.py`) — appears in the Analytics comparison table when trained
42. Optional Google Fact Check API integration — shows related professional fact-checks under a verdict
43. Account management — change password, export full history as CSV, delete individual history entries
44. Client-side history search/filter on the dashboard
45. Toast-style animated flash messages
46. Copy-result-as-text button on the Detect page
47. MIT License + favicon + Open Graph meta tags for a polished, shareable repo

That's **47 distinct, working capabilities** across 6 categories — built for
depth without unnecessary bloat, and everything above is either directly
testable in the app or verifiable in `test_app.py`.

---

## Project structure

```
VeriFactPro/
├── app.py                     # Flask app — routes, auth, database, all APIs
├── make_admin.py              # CLI helper to promote a user to admin
├── test_app.py                # pytest suite (18 tests)
├── requirements.txt / requirements-dev.txt
├── Dockerfile / docker-compose.yml
├── Procfile / runtime.txt     # for Render/Railway deployment
├── .env.example                # copy to .env for local secrets
├── LICENSE                    # MIT
├── README.md
├── .github/workflows/tests.yml  # CI — runs tests on every push
├── browser-extension/         # companion Chrome extension
├── data/
│   ├── README.md              # dataset instructions (Kaggle link + fallback)
│   └── generate_sample_data.py
├── model/
│   ├── preprocess.py          # shared text-cleaning function
│   ├── explain.py             # per-request + global explainability
│   ├── factcheck.py           # optional Google Fact Check API integration
│   ├── train_model.py         # trains + evaluates + saves the model + charts
│   ├── train_bert.py          # optional DistilBERT comparison model
│   ├── model.pkl / vectorizer.pkl / metrics.json   # generated after training
├── templates/                 # 13 pages
└── static/
    ├── css/style.css          # dark + light theme, all components
    ├── js/                    # script.js, compare.js, quiz.js, theme.js
    └── img/                   # charts + word clouds, generated after training
```

## Pages

| Page | Route | Login required? |
|---|---|---|
| Home | `/` | No |
| Detect (text / URL / batch) | `/detect` | No |
| Compare | `/compare` | No |
| How it works | `/about` | No |
| Analytics | `/analytics` | No |
| Quiz | `/quiz` | No |
| FAQ | `/faq` | No |
| Sign up / Log in | `/signup`, `/login` | No |
| Dashboard (your history) | `/dashboard` | Yes |
| Admin panel | `/admin` | Yes (admin only) |

---

## Setup (VS Code, local, CPU only)

### 1. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a dataset
This project can combine multiple datasets automatically — just drop the
files in `data/` in any combination and training picks them all up.

**Recommended (biggest, most robust): ISOT + WELFake (~117,000 articles combined)**
- ISOT: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset → `data/Fake.csv` + `data/True.csv`
- WELFake: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification → `data/WELFake_Dataset.csv`

Full instructions, an optional third dataset (LIAR), and the demo-data
fallback are in `data/README.md`.

**Or, instant demo dataset (no download needed):**
```bash
python data/generate_sample_data.py
```

### 4. Train the model
```bash
python model/train_model.py
```
This also generates the word clouds used on the Analytics page. Takes ~1-3
minutes on CPU with the real dataset.

### 5. Run the website
```bash
python app.py
```
Open **http://127.0.0.1:5000**

### 6. (Optional) Make yourself an admin
```bash
# First sign up on the website normally, then:
python make_admin.py your_username
```
Log back in and visit `/admin`.

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest test_app.py -v
```
18 tests covering every page, signup/login/duplicate handling, and the
prediction API. CI (`.github/workflows/tests.yml`) runs these automatically
on every push to GitHub.

## Running with Docker (alternative to the venv setup above)

```bash
docker compose up --build
```
Then open **http://127.0.0.1:5000**. This builds a container with all
dependencies pinned, so it behaves identically on any machine — useful for
showing deployment/DevOps awareness in a viva.
(You still need to train the model first, or mount a pre-trained `model/`
folder — see `docker-compose.yml`.)

## Configuration (.env)

Copy `.env.example` to `.env` and fill in real values:
```bash
cp .env.example .env      # Mac/Linux
copy .env.example .env    # Windows
```
`SECRET_KEY` should be a random string in any real deployment.
`FACTCHECK_API_KEY` is optional (see below).

---

## API usage example

Every registered user gets an API key on their dashboard. Example request:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Officials confirmed the policy will proceed after review..."}'
```

Response:
```json
{
  "prediction": "REAL",
  "confidence": 92.3,
  "real_probability": 92.3,
  "fake_probability": 7.7,
  "model_used": "Logistic Regression",
  "fake_indicator_words": [],
  "real_indicator_words": ["officials", "committee", "review"]
}
```

---

## Tech stack

- **ML/NLP:** scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, Passive Aggressive), NLTK
- **Backend:** Flask, Flask-Login, Flask-SQLAlchemy, SQLite
- **Scraping:** requests + BeautifulSoup (for URL detection)
- **PDF generation:** ReportLab
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Frontend:** HTML/CSS/JS (vanilla, no build step), Jinja2 templates, Web Speech API (voice input)

## Optional: Fact-check cross-referencing

The Detect page can show related professional fact-checks (via Google's
free Fact Check Tools API) alongside a verdict. This is optional and off
by default — without an API key, the app works exactly as before, just
without this section.

To enable:
1. Go to https://console.cloud.google.com/, create a project (free).
2. Enable the "Fact Check Tools API".
3. Create an API key under APIs & Services → Credentials.
4. Before running the app: `set FACTCHECK_API_KEY=your_key` (Windows) or
   `export FACTCHECK_API_KEY=your_key` (Mac/Linux).

## Optional: Browser extension

`browser-extension/` contains a minimal Chrome extension that lets you
right-click any selected text on any webpage → "Check with VeriFact" →
get an instant verdict, using your local Flask app as the backend.

**To load it:**
1. Make sure `python app.py` is running (the extension calls `127.0.0.1:5000`).
2. Open Chrome → `chrome://extensions` → enable "Developer mode" (top right).
3. Click "Load unpacked" → select the `browser-extension` folder.
4. Select any text on any webpage → right-click → "Check with VeriFact:..."

## Optional: Deep learning comparison (DistilBERT)

The deployed model is TF-IDF + classic ML by design (fast, CPU-friendly,
explainable). If you also want to show deep learning skills, `model/train_bert.py`
fine-tunes a DistilBERT classifier as a *separate* comparison model and
automatically adds its accuracy to the Analytics page's comparison table.

```bash
pip install torch transformers
python model/train_bert.py
```

This needs real compute time (a few hours on CPU, ~15-20 min on a free
Colab/Kaggle GPU) and is entirely optional — the website only ever uses
`model/model.pkl`, so skipping this doesn't affect anything else.

## Recording a demo for your README/LinkedIn

A short screen-recording turns your GitHub repo into something people
actually watch instead of skim. Free tools: ScreenToGif (Windows),
Kap (Mac), or just Xbox Game Bar (Win+G) exported to GIF/MP4. Record
~20-30 seconds: paste an article → show the verdict → flip to Analytics.
Drop the file in the repo and reference it at the top of this README:
```markdown
![Demo](demo.gif)
```

## Explainability — how it works

For linear models (Logistic Regression), each word in the TF-IDF vocabulary
has a learned weight. Multiplying a document's TF-IDF score for a word by
that word's weight shows how much that word pushed the prediction toward
FAKE or REAL. `model/explain.py` implements this and the Detect page
displays the top contributing words as colored chips under each result.

## Optional: going further with BERT

The deployed model is TF-IDF + classic ML by design — fast, CPU-friendly,
and easy to explain in a viva. If you want to additionally showcase deep
learning skills, you can train a DistilBERT classifier as a *separate*
comparison model (requires `transformers` + `torch`, a few hours on CPU or
minutes on GPU/Colab) and add its accuracy to the Analytics page's model
comparison table. This is intentionally left out of the default setup to
keep training fast and dependency-light, matching the "no GPU needed" brief.

## Limitations (worth stating honestly in your report)

- The model detects stylistic patterns correlated with fake vs. real
  articles in its training data — it does not fact-check claims.
- URL detection depends on simple HTML scraping; JavaScript-rendered sites
  or sites that block bots may not extract cleanly.
- Performance reflects the training distribution (ISOT is US politics/news-heavy).
- This is a decision-support signal, not a ground-truth verifier.
