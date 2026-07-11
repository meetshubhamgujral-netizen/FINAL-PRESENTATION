# 🧭 BridgeCompliance Advisory — Market Validation Dashboard

Interactive Streamlit dashboard that turns a synthetic B2B compliance-advisory
survey (n = 1,190) into **descriptive, predictive and prescriptive** insight for
a cross-border legal advisory firm targeting tech/fintech startups in **Dubai
(DIFC/ADGM)** and **Singapore (MAS)**.

Author: **Angie Ximena Lozano Rincón** — Data Analytics, Term 2, SP Jain Global
School of Management.

---

## What it answers

1. **Which market / customer to target?**
2. **What is their buying capacity (willingness to pay)?**
3. **How do they buy (behaviour / engagement)?**
4. **Which product/service should we recommend?**

## The seven tabs (decision-first)

The dashboard opens with the answer and works down to the proof, so a non-analyst reads it top to bottom. Each tab is labelled with its analytics type.

| Tab | Type | What happens |
|-----|------|--------------|
| **0 · Executive Verdict** | Strategic Summary | Answer cards, a five-step decision cascade, and a base-case revenue opportunity |
| **1 · Objectives** | Overview | Business context, the four commercial questions, the analytics ladder, the honesty note, and a plain-English explanation of what a "chain" of models is |
| **2 · Is the Market Real?** | Descriptive & Diagnostic | Market head-to-head KPIs, WTP by stage, intent by stage×market, pain prevalence, urgency violins, **significance tests** (chi-square / Mann-Whitney / Kruskal-Wallis), Spearman matrix + WTP drivers |
| **3 · Who to Target** | Predictive | Leakage honesty test, **2×2 targeting map** (want vs pay), **Classifier Chains** per-link accuracy, and what drives priority — all leakage-free |
| **4 · The Money** | Prescriptive | A **scenario revenue model** you drive (TAM, fit %, conversion, prices) with live ARR, tier breakdown and currency toggle — explicitly not a forecast |
| **5 · The Playbook** | Prescriptive | K-Prototypes segments (table + radar + sizes), Apriori service bundles (rules + bubble), pricing tiers, expected-value lead list, and a month-by-month GTM sequence |
| **6 · Methodology** | Technical Appendix | Model comparison, ROC, confusion matrix, CV, leakage demo, Ridge/Lasso regression, clustering elbow, data-sufficiency plan and production monitoring — all in expanders |

Every analytical tab follows the same rhythm: **a story intro tying the analysis to the market-entry strategy → a meticulous "why this method (and not the alternative)" → charts/tables → a plain-English takeaway.** A branded sidebar carries a **currency toggle (SGD / AED / USD)** and Market / Stage / Industry filters that drive the Market and Playbook views (predictive models always train on the full sample for stability).

## A note on honesty (important for the defence)

The dataset is **synthetic**, and the target labels (High Intent, High WTP,
Priority) were **derived by thresholding the survey features** during
generation. That is *target leakage*. This app therefore reports predictive
performance on a **leakage-free feature set** and explicitly **demonstrates the
trap** (accuracy jumps to ~100% the moment the label's source feature is added
back). `Priority ≈ High_Intent AND High_WTP` (≈99% of rows) — which is why the
targets form a **chain** and Classifier Chains is the right model.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset ships **bundled** in the repo — no upload needed. The app finds
`BridgeCompliance_Preprocessed_Dataset.xlsx` whether it sits at the repo root,
in a `data/` folder, or anywhere in the project.

## Deploy on Streamlit Community Cloud

1. Create a **new GitHub repo** and upload **all the files in this folder** (they
   are all flat — select them and drag them in together, not the folder itself).
2. Go to <https://share.streamlit.io> → **Create app** → **Deploy from GitHub**.
3. Pick your repo, branch `main`, **Main file path: `app.py`**.
4. **IMPORTANT — open `Advanced settings` and set Python version to `3.12`**
   before clicking Deploy. Streamlit Cloud may default to Python 3.14, on which
   parts of the scientific stack still fail at runtime. Python 3.12 is stable for
   the whole stack (scikit-learn, scipy, kmodes, mlxtend, plotly).
5. **Deploy.** First load takes ~2–3 min (it trains the models once, then caches).

If the app ever shows "Oh no. Error running app." it is almost always the Python
version — delete the app and redeploy choosing **Python 3.12** in Advanced settings.

## Files

```
app.py                                   # Streamlit UI — the 4 tabs
analysis.py                              # Pure ML/analytics logic (runs standalone)
requirements.txt
README.md
.gitignore
BridgeCompliance_Preprocessed_Dataset.xlsx   # the bundled dataset (at root)
```

## Methodology references

Read et al. (2011), *Machine Learning* 85(3) — Classifier Chains · Huang (1998) —
K-Prototypes · Rousseeuw (1987) — Silhouette · Agrawal & Srikant (1994) — Apriori
· James et al. (2013), *ISLR* — cross-validation · Cawley & Talbot (2010), *JMLR* —
nested CV / model-selection bias · Elkan (2001) — cost-sensitive learning.
