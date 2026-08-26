# 🛡️ TruthGuard — AI Powered Fake News Detection System

TruthGuard is a full-stack Flask web application that uses Natural Language
Processing (NLP) and classical Machine Learning to predict whether a piece
of news text is **Fake** or **Real**, complete with a confidence score,
user accounts, prediction history, and a modern glassmorphism UI.

> ⚠️ **Educational project.** Predictions reflect statistical patterns learned
> from training data and are not a substitute for professional fact-checking.

---
## 🚀 Live Demo

👉 [Try TruthGuard Live](https://truthguard-ai-powered-fake-news-detection.onrender.com)

## 🚀 Git_Repo
👉 [View Git_Repo Code](https://github.com/TarunSonaji/TruthGuard-AI-powerded-Fake-News-Detection)

## 1. Project Overview

| | |
|---|---|
| **Backend** | Python, Flask |
| **ML** | Scikit-learn (TF-IDF + Passive Aggressive / Logistic Regression / Naive Bayes / Linear SVM) |
| **NLP** | NLTK (tokenization, stop words, lemmatization) |
| **Database** | SQLite via SQLAlchemy (Users + Predictions) |
| **Auth** | Flask-Login with hashed passwords |
| **Frontend** | HTML5, CSS3, Bootstrap 5, vanilla JavaScript |

The repo already ships with a **pre-trained demo model** (`model.pkl` /
`vectorizer.pkl`) trained on a small synthetic dataset, so you can run the
app immediately. For real-world accuracy, retrain on the full Kaggle dataset
(see Section 3).

---

## 2. Installation

```bash
# 1. Clone / unzip the project, then move into it
cd FakeNewsDetection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (First time only) Download NLTK corpora
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

> If you're offline or NLTK downloads fail, the app automatically falls back
> to a built-in stop word list so everything still works — you just lose
> lemmatization quality slightly.

---

## 3. Dataset Instructions

TruthGuard is designed around the Kaggle **"Fake and Real News Dataset"**:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

1. Download `Fake.csv` and `True.csv` from Kaggle.
2. Place both files inside the `dataset/` folder (same filenames).
3. Run the training script (Section 4).

**Don't have a Kaggle account / want to test the app right now?**
A synthetic dataset generator is included:

```bash
python dataset/generate_sample_dataset.py
```

This creates small demo `Fake.csv` / `True.csv` files with the same column
structure, so the entire pipeline works end-to-end without any download.
(This is also what the pre-shipped `model.pkl` in this repo was trained on
— swap in the real Kaggle CSVs and retrain for production-quality results.)

---

## 4. Training the Model

```bash
python train_model.py
```

This will:
1. Load and label `dataset/Fake.csv` (0) and `dataset/True.csv` (1)
2. Clean the text (lowercase, remove URLs/punctuation/numbers, remove stop
   words, tokenize, lemmatize)
3. Vectorize with `TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")`
4. Train **4 models**: Passive Aggressive Classifier, Logistic Regression,
   Multinomial Naive Bayes, Linear SVM
5. Evaluate each on accuracy / precision / recall / F1 + confusion matrix
6. Save the **best model** (by F1) to `model.pkl`, the vectorizer to
   `vectorizer.pkl`, and full metrics to `model_metrics.json` (shown on the
   `/about` page)

---

## 5. Running the Flask Server

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

Register an account, log in, and paste any headline or article into the
textarea on the home page to get an instant Fake/Real prediction.

---

## 6. Folder Structure

```
FakeNewsDetection/
│
├── app.py                     # Flask app factory + all routes
├── train_model.py             # ML training pipeline
├── predict.py                 # Loads model/vectorizer, runs predictions
├── config.py                  # App configuration
├── database.py                # SQLAlchemy + Flask-Login setup
├── models.py                  # User & Prediction ORM models
├── requirements.txt
├── model.pkl                  # Trained classifier (generated)
├── vectorizer.pkl             # Fitted TF-IDF vectorizer (generated)
├── model_metrics.json         # Training metrics (generated)
│
├── utils/
│   └── text_preprocessing.py  # Shared NLP cleaning pipeline
│
├── dataset/
│   ├── Fake.csv                       # (you provide / generate)
│   ├── True.csv                       # (you provide / generate)
│   └── generate_sample_dataset.py     # synthetic demo data generator
│
├── templates/
│   ├── base.html, index.html, login.html, register.html,
│   │   history.html, about.html, profile.html, 404.html, 500.html
│
├── static/
│   ├── css/style.css          # Glassmorphism / gradient / dark mode UI
│   ├── js/main.js             # AJAX prediction, toasts, theme toggle
│   ├── images/                # (screenshots, icons)
│   └── uploads/               # (report exports)
│
└── instance/
    └── truthguard.db          # SQLite database (auto-created)
```

---

## 7. Features

- ✅ Fake / Real prediction with confidence score
- ✅ Suspicious word/phrase highlighting for flagged articles
- ✅ User registration, login, logout (hashed passwords, sessions)
- ✅ Per-user prediction history with search + CSV export
- ✅ Public JSON REST API: `POST /api/predict`
- ✅ Responsive glassmorphism UI with dark mode toggle
- ✅ Toast notifications, progress bar, character counter, copy/download report
- ✅ Model comparison dashboard on `/about`

### REST API Example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover shocking secret government cover-up!!!"}'
```

```json
{
  "success": true,
  "prediction": "Fake",
  "confidence": 82.4,
  "suspicious_words": ["shocking", "secret", "cover"]
}
```

---

## 8. Screenshots

*(Add screenshots here after running the app)*

- `static/images/home.png` — Home page / analyzer
- `static/images/result.png` — Prediction result card
- `static/images/history.png` — History dashboard
- `static/images/about.png` — Model comparison table

---

## 9. Future Enhancements

- Swap classical ML for a fine-tuned transformer (BERT / DistilBERT) for higher accuracy
- Add an admin dashboard for monitoring usage across all users
- Multi-language fake news detection
- Browser extension for one-click checking while reading news sites
- Source credibility scoring (cross-reference the publishing domain)
- Dockerfile + docker-compose for one-command deployment

---

## 10. Disclaimer

This tool is built for educational and demonstration purposes. It should
never be used as the sole basis for determining the truthfulness of news.
Always cross-check with reputable, independent fact-checking sources.
