"""
analysis.py — Pure analytical/ML logic for the BridgeCompliance dashboard.

No Streamlit here on purpose: every function runs and unit-tests standalone
(see __main__). app.py imports these and wraps them with caching + UI.

Honesty principle: the three binary targets (High_Intent, High_WTP, Priority)
were derived by thresholding survey features during synthetic generation
(Intent from Q8_Likelihood, WTP from the price tier, Priority ~= Intent AND WTP).
That is target leakage, so all predictive numbers use a LEAKAGE-FREE feature
set; we also expose the leaked version purely to demonstrate the trap.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, chi2_contingency, mannwhitneyu, kruskal
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import (
    cross_val_score, train_test_split, cross_val_predict, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Column groups                                                               #
# --------------------------------------------------------------------------- #
HONEST_FEATURES = [
    "ENC_Market", "ENC_Stage",
    "Q4_Licensing", "Q4_AMLCFT", "Q4_DataProtection",
    "Q4_DigitalAssets", "Q4_AIGovernance", "Q4_Pain_Count",
    "Q5_Urgency", "LE_Q2_Industry", "LE_Q6_Engagement", "LE_Q9_Deal_Breaker",
]
LEAK_FEATURE_INTENT = "Q8_Likelihood"
TARGETS = ["TARGET_High_Intent", "TARGET_High_WTP", "TARGET_Priority"]
TARGET_PRETTY = {
    "TARGET_High_Intent": "High Intent (wants it?)",
    "TARGET_High_WTP": "High Willingness to Pay (can pay?)",
    "TARGET_Priority": "Priority Client (chase now?)",
}
PAIN_COLS = {
    "Q4_Licensing": "Licensing", "Q4_AMLCFT": "AML/CFT",
    "Q4_DataProtection": "Data Protection", "Q4_DigitalAssets": "Digital Assets",
    "Q4_AIGovernance": "AI Governance",
}
FEATURE_PRETTY = {
    "Q5_Urgency": "Urgency", "ENC_Stage": "Funding stage",
    "Q4_Pain_Count": "Pain count", "LE_Q2_Industry": "Industry",
    "LE_Q9_Deal_Breaker": "Deal-breaker", "LE_Q6_Engagement": "Engagement model",
    "ENC_Market": "Market", "Q4_Licensing": "Pain: Licensing",
    "Q4_AMLCFT": "Pain: AML/CFT", "Q4_DataProtection": "Pain: Data",
    "Q4_DigitalAssets": "Pain: Digital assets", "Q4_AIGovernance": "Pain: AI Gov.",
}
STAGE_ORDER = ["Pre-seed (1-5 ppl)", "Seed (6-20 ppl)",
               "Series A-B (21-100 ppl)", "Series C+ (100+ ppl)"]


def load_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Clean Dataset", header=1)
    if "BOT_Flag" in df.columns:
        df = df[df["BOT_Flag"] == 0].copy()
    return df.reset_index(drop=True)


# =========================================================================== #
# TAB — DESCRIPTIVE & DIAGNOSTIC                                               #
# =========================================================================== #
def market_summary(df):
    return df.groupby("Q1_Market").agg(
        Intent_Rate=("TARGET_High_Intent", "mean"),
        Avg_WTP_SGD=("ENC_WTP_SGD", "mean"),
        Avg_Urgency=("Q5_Urgency", "mean"),
        Avg_Likelihood=("Q8_Likelihood", "mean"),
        N=("ResponseID", "count"),
    ).round(2)


def wtp_by_stage(df):
    return df.groupby("Q3_Stage")["ENC_WTP_SGD"].mean().reindex(STAGE_ORDER).round(0)


def intent_by_stage_market(df):
    g = df.groupby(["Q3_Stage", "Q1_Market"])["TARGET_High_Intent"].mean().reset_index()
    g["Q3_Stage"] = pd.Categorical(g["Q3_Stage"], STAGE_ORDER, ordered=True)
    return g.sort_values("Q3_Stage")


def pain_prevalence(df):
    rows = []
    for col, name in PAIN_COLS.items():
        for mk in df["Q1_Market"].unique():
            sub = df[df["Q1_Market"] == mk]
            rows.append({"Pain": name, "Market": mk, "Prevalence": sub[col].mean()})
    return pd.DataFrame(rows)


def urgency_by_market(df):
    return df[["Q1_Market", "Q5_Urgency", "Q8_Likelihood"]].copy()


def dealbreaker_stacked(df):
    ct = pd.crosstab(df["Q1_Market"], df["Q9_Deal_Breaker"], normalize="index")
    return ct.reset_index().melt(id_vars="Q1_Market", var_name="Factor", value_name="Share")


def engagement_mix(df):
    return (df["Q6_Engagement"].value_counts(normalize=True)
            .rename_axis("Model").reset_index(name="Share"))


def spearman_vs_wtp(df):
    drivers = {
        "Stage": "ENC_Stage", "Urgency": "Q5_Urgency", "Likelihood": "Q8_Likelihood",
        "Pain count": "Q4_Pain_Count", "Market (Dubai=1)": "ENC_Market",
        "Pain: Licensing": "Q4_Licensing", "Pain: AML/CFT": "Q4_AMLCFT",
        "Pain: Data": "Q4_DataProtection", "Pain: Digital": "Q4_DigitalAssets",
        "Pain: AI Gov.": "Q4_AIGovernance",
    }
    rows = [{"Driver": n, "rho": round(spearmanr(df[c], df["ENC_WTP_SGD"])[0], 3)}
            for n, c in drivers.items()]
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)


def spearman_matrix(df):
    cols = {
        "WTP": "ENC_WTP_SGD", "Stage": "ENC_Stage", "Urgency": "Q5_Urgency",
        "Likelihood": "Q8_Likelihood", "Pain count": "Q4_Pain_Count",
        "Licensing": "Q4_Licensing", "AML/CFT": "Q4_AMLCFT", "Data": "Q4_DataProtection",
        "Digital": "Q4_DigitalAssets", "AI Gov.": "Q4_AIGovernance",
    }
    sub = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    return sub.corr(method="spearman").round(2)


def scatter_stage_wtp(df):
    rng = np.random.default_rng(RANDOM_STATE)
    out = df[["ENC_Stage", "ENC_WTP_SGD", "Q1_Market", "Q3_Stage"]].copy()
    out["stage_jit"] = out["ENC_Stage"] + rng.normal(0, 0.06, len(out))
    out["wtp_jit"] = out["ENC_WTP_SGD"] + rng.normal(0, 120, len(out))
    return out


def significance_tests(df):
    """Formal tests behind the 'the city is not the lever' claim."""
    out = {}
    ct = pd.crosstab(df["Q1_Market"], df["TARGET_High_Intent"])
    chi2, p, _, _ = chi2_contingency(ct)
    out["Intent rate: Dubai vs Singapore"] = {
        "Test": "Chi-square", "Statistic": round(chi2, 2), "p-value": round(p, 3),
        "Significant?": "No" if p >= 0.05 else "Yes"}
    grps = [d for _, d in df.groupby("Q1_Market")]
    if len(grps) == 2:
        for label, col in [("Willingness to pay: Dubai vs Singapore", "ENC_WTP_SGD"),
                           ("Urgency: Dubai vs Singapore", "Q5_Urgency")]:
            u, p = mannwhitneyu(grps[0][col], grps[1][col])
            out[label] = {"Test": "Mann-Whitney", "Statistic": round(u, 0),
                          "p-value": round(p, 3), "Significant?": "No" if p >= 0.05 else "Yes"}
    h, p = kruskal(*[d["ENC_WTP_SGD"].values for _, d in df.groupby("Q3_Stage")])
    out["Willingness to pay across stages"] = {
        "Test": "Kruskal-Wallis", "Statistic": round(h, 1), "p-value": round(p, 4),
        "Significant?": "No" if p >= 0.05 else "Yes"}
    return pd.DataFrame(out).T.reset_index().rename(columns={"index": "Comparison"})


def quadrant_df(df):
    """Per-startup Intent (x) vs Willingness to pay (y) for the 2x2 targeting map."""
    rng = np.random.default_rng(RANDOM_STATE)
    d = df[["Q8_Likelihood", "ENC_WTP_SGD", "Q2_Industry", "Q3_Stage",
            "TARGET_Priority"]].copy()
    d["intent_jit"] = d["Q8_Likelihood"] + rng.normal(0, 0.18, len(d))
    d["wtp_jit"] = d["ENC_WTP_SGD"] + rng.normal(0, 260, len(d))
    d["Segment"] = np.where(d["TARGET_Priority"] == 1, "Priority (wants & pays)", "Other")
    return d, float(d["Q8_Likelihood"].median()), float(d["ENC_WTP_SGD"].median())


def wtp_tier_shares(df):
    vc = df["Q7_WTP_Tier"].value_counts(normalize=True)
    out = {"Early": 0.0, "Growth": 0.0, "Scale": 0.0}
    for k, v in vc.items():
        if "Early" in k:
            out["Early"] = round(float(v), 3)
        elif "Growth" in k:
            out["Growth"] = round(float(v), 3)
        elif "Scale" in k:
            out["Scale"] = round(float(v), 3)
    return out


def revenue_model(tam, pct_fit, conversion, prices, shares, contract_months=12):
    """Scenario revenue math. Data supplies the SHAPE (tier mix); the user supplies
    the SCALE (TAM, fit %, conversion). Nothing here is a prediction."""
    clients = tam * pct_fit * conversion
    rows = []
    for tier, key in [("Starter", "Early"), ("Growth", "Growth"), ("Scale", "Scale")]:
        cl = clients * shares.get(key, 0.0)
        arr = cl * prices[tier] * contract_months
        rows.append({"Tier": tier, "Clients": round(cl, 1),
                     "Price (SGD/mo)": prices[tier], "ARR (SGD)": round(arr)})
    dfm = pd.DataFrame(rows)
    return dfm, int(dfm["ARR (SGD)"].sum()), clients


# =========================================================================== #
# TAB — PREDICTIVE                                                             #
# =========================================================================== #
def _models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Logistic Reg.": make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight="balanced")),
    }


def honest_classification_scores(df):
    X = df[HONEST_FEATURES]
    rows = []
    for t in TARGETS:
        y = df[t]
        base = max(y.mean(), 1 - y.mean())
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        acc = cross_val_score(gb, X, y, cv=5, scoring="accuracy").mean()
        au = cross_val_score(gb, X, y, cv=5, scoring="roc_auc").mean()
        rows.append({"Target": TARGET_PRETTY[t], "Baseline (chance)": round(base, 3),
                     "Honest accuracy": round(acc, 3), "Honest AUC": round(au, 3)})
    return pd.DataFrame(rows)


def leakage_demonstration(df):
    y = df["TARGET_High_Intent"]
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    honest = cross_val_score(gb, df[HONEST_FEATURES], y, cv=5, scoring="accuracy").mean()
    leak = cross_val_score(gb, df[HONEST_FEATURES + [LEAK_FEATURE_INTENT]], y,
                           cv=5, scoring="accuracy").mean()
    return {"honest_acc": round(honest, 3), "leaked_acc": round(leak, 3),
            "baseline": round(max(y.mean(), 1 - y.mean()), 3)}


def model_comparison(df, target="TARGET_Priority"):
    X, y = df[HONEST_FEATURES], df[target]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, m in _models().items():
        rows.append({
            "Model": name,
            "Accuracy": round(cross_val_score(m, X, y, cv=cv, scoring="accuracy").mean(), 3),
            "Precision": round(cross_val_score(m, X, y, cv=cv, scoring="precision").mean(), 3),
            "Recall *": round(cross_val_score(m, X, y, cv=cv, scoring="recall").mean(), 3),
            "F1": round(cross_val_score(m, X, y, cv=cv, scoring="f1").mean(), 3),
            "AUC": round(cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean(), 3),
        })
    return pd.DataFrame(rows)


def roc_data(df, target="TARGET_Priority"):
    X, y = df[HONEST_FEATURES], df[target].values
    out = {}
    for name, m in _models().items():
        proba = cross_val_predict(m, X, y, cv=5, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        out[name] = {"fpr": fpr, "tpr": tpr, "auc": round(auc(fpr, tpr), 3)}
    return out


def confusion_matrices(df):
    X = df[HONEST_FEATURES]
    out = {}
    for t in TARGETS:
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        pred = cross_val_predict(gb, X, df[t], cv=5)
        out[TARGET_PRETTY[t]] = confusion_matrix(df[t], pred)
    return out


def cv_scores(df, target="TARGET_Priority"):
    X, y = df[HONEST_FEATURES], df[target]
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return cross_val_score(gb, X, y, cv=5, scoring="f1")


def feature_importance_rf_gb(df, target="TARGET_Priority"):
    X, y = df[HONEST_FEATURES], df[target]
    rf = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE).fit(X, y)
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE).fit(X, y)
    d = pd.DataFrame({"Feature": [FEATURE_PRETTY.get(f, f) for f in HONEST_FEATURES],
                      "Random Forest": rf.feature_importances_,
                      "Gradient Boosting": gb.feature_importances_})
    d = d.sort_values("Gradient Boosting", ascending=False).reset_index(drop=True)
    return d.melt(id_vars="Feature", var_name="Model", value_name="Importance")


def fit_classifier_chain(df):
    X, Y = df[HONEST_FEATURES].values, df[TARGETS].values
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3,
                                          random_state=RANDOM_STATE, stratify=Y[:, 2])
    chain = ClassifierChain(GradientBoostingClassifier(random_state=RANDOM_STATE),
                            order=[0, 1, 2], random_state=RANDOM_STATE).fit(Xtr, Ytr)
    Yp = chain.predict(Xte)
    per = {TARGET_PRETTY[t]: round((Yp[:, i] == Yte[:, i]).mean(), 3)
           for i, t in enumerate(TARGETS)}
    return {"per_label_acc": per, "priority_cm": confusion_matrix(Yte[:, 2], Yp[:, 2])}


def wtp_regression(df):
    X, y = df[HONEST_FEATURES].values, df["ENC_WTP_SGD"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    metrics, coefs = {}, {}
    for name, model in [("Linear", LinearRegression()),
                        ("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=50.0))]:
        pipe = make_pipeline(StandardScaler(), model).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        metrics[name] = {"R2_test": round(r2_score(yte, pred), 3),
                         "MAE_SGD": int(mean_absolute_error(yte, pred))}
        coefs[name] = pipe.steps[-1][1].coef_
    cf = pd.DataFrame({"Feature": [FEATURE_PRETTY.get(f, f) for f in HONEST_FEATURES],
                       "Ridge": coefs["Ridge"], "Lasso": coefs["Lasso"]})
    cf["abs"] = cf["Lasso"].abs()
    cf = cf.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr)
    pred = ridge.predict(Xte)
    return {"metrics": metrics, "coefs": cf, "actual": yte, "pred": pred,
            "resid": yte - pred}


# =========================================================================== #
# TAB — PRESCRIPTIVE                                                           #
# =========================================================================== #
_NUM = ["Q5_Urgency", "Q8_Likelihood", "ENC_WTP_SGD", "ENC_Stage", "Q4_Pain_Count"]
_CAT = ["Q1_Market", "Q2_Industry", "Q6_Engagement", "Q9_Deal_Breaker"]


def _cluster_labels(df, k):
    work = df[_NUM + _CAT].copy()
    try:
        from kmodes.kprototypes import KPrototypes
        mat = work.copy()
        mat[_NUM] = StandardScaler().fit_transform(mat[_NUM])
        cat_idx = [work.columns.get_loc(c) for c in _CAT]
        kp = KPrototypes(n_clusters=k, init="Huang", random_state=RANDOM_STATE, n_init=3)
        labels = kp.fit_predict(mat.to_numpy(), categorical=cat_idx)
        cost = kp.cost_
    except Exception:
        from sklearn.cluster import KMeans
        X = np.hstack([StandardScaler().fit_transform(work[_NUM]),
                       pd.get_dummies(work[_CAT]).to_numpy()])
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        labels, cost = km.labels_, km.inertia_
    return labels, cost


def cluster_cost_curve(df, kmax=6):
    return pd.DataFrame([{"k": k, "cost": _cluster_labels(df, k)[1]}
                         for k in range(2, kmax + 1)])


def run_clustering(df, k=4):
    labels, _ = _cluster_labels(df, k)
    tmp = df.copy()
    tmp["Segment"] = labels
    prof = tmp.groupby("Segment").agg(
        Urgency=("Q5_Urgency", "mean"), Likelihood=("Q8_Likelihood", "mean"),
        WTP_SGD=("ENC_WTP_SGD", "mean"), Stage=("ENC_Stage", "mean"),
        Pain_Count=("Q4_Pain_Count", "mean"), Priority_Rate=("TARGET_Priority", "mean"),
        N=("ResponseID", "count"),
    ).round(2)
    dubai = (tmp[tmp["Q1_Market"].str.contains("Dubai")].groupby("Segment").size()
             / tmp.groupby("Segment").size()).round(2)
    prof["Dubai_%"] = (dubai * 100).round(0)
    stage_mode = tmp.groupby("Segment")["Q3_Stage"].agg(lambda s: s.mode().iat[0])
    prof["StageLabel"] = stage_mode.map(lambda x: x.split(" (")[0])
    # Name by descending priority: the segment most likely to convert leads first.
    prof = prof.sort_values("Priority_Rate", ascending=False)
    names = ["Priority Converter", "Emerging Buyer", "Slow Burner", "Explorer"]
    prof.insert(0, "Profile", [names[i] if i < len(names) else f"Segment {i}"
                               for i in range(len(prof))])
    return prof.reset_index()


def association_rules_pain(df, min_support=0.15, min_lift=1.0):
    from mlxtend.frequent_patterns import apriori, association_rules
    basket = df[list(PAIN_COLS)].rename(columns=PAIN_COLS).astype(bool)
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame(columns=["IF (has)", "THEN (also needs)",
                                     "Support", "Confidence", "Lift"])
    r = association_rules(freq, metric="lift", min_threshold=min_lift)
    r = r[r["lift"] > min_lift].sort_values("lift", ascending=False)
    fmt = lambda s: " + ".join(sorted(list(s)))
    return pd.DataFrame({
        "IF (has)": r["antecedents"].apply(fmt),
        "THEN (also needs)": r["consequents"].apply(fmt),
        "Support": r["support"].round(3), "Confidence": r["confidence"].round(3),
        "Lift": r["lift"].round(3)}).reset_index(drop=True).head(12)


def lead_ranking(df, top_n=15):
    X, y = df[HONEST_FEATURES], df["TARGET_Priority"]
    proba = cross_val_predict(GradientBoostingClassifier(random_state=RANDOM_STATE),
                              X, y, cv=5, method="predict_proba")[:, 1]
    out = df[["ResponseID", "Q1_Market", "Q2_Industry", "Q3_Stage",
              "ENC_WTP_SGD", "Q5_Urgency"]].copy()
    out["P_Priority"] = proba.round(3)
    out["Expected_Value_SGD"] = (proba * out["ENC_WTP_SGD"]).round(0)
    out = out.sort_values("Expected_Value_SGD", ascending=False).head(top_n).reset_index(drop=True)
    return out.rename(columns={"Q1_Market": "Market", "Q2_Industry": "Industry",
                               "Q3_Stage": "Stage", "ENC_WTP_SGD": "WTP (SGD/mo)",
                               "Q5_Urgency": "Urgency"})


def pricing_tiers():
    return pd.DataFrame([
        {"Tier": "Starter", "Stage": "Pre-seed / Seed", "WTP (SGD/mo)": "800 - 2,000",
         "Recommended package": "One-off regulatory audit + licensing roadmap"},
        {"Tier": "Growth", "Stage": "Series A-B", "WTP (SGD/mo)": "2,000 - 5,000",
         "Recommended package": "Monthly retainer - AML framework + regulatory monitoring"},
        {"Tier": "Scale", "Stage": "Series C+", "WTP (SGD/mo)": "5,000 - 10,000",
         "Recommended package": "Annual contract - dual-jurisdiction compliance"},
    ])


# =========================================================================== #
if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    name = "BridgeCompliance_Preprocessed_Dataset.xlsx"
    path = next((p for p in [os.path.join(here, name), os.path.join(here, "data", name)]
                 if os.path.exists(p)), os.path.join(here, name))
    df = load_data(path)
    print("rows:", len(df))
    for fn in [market_summary, wtp_by_stage, intent_by_stage_market, pain_prevalence,
               dealbreaker_stacked, engagement_mix, spearman_vs_wtp, spearman_matrix,
               scatter_stage_wtp, honest_classification_scores, leakage_demonstration,
               model_comparison, confusion_matrices, feature_importance_rf_gb,
               fit_classifier_chain, wtp_regression, cluster_cost_curve, run_clustering,
               association_rules_pain, lead_ranking]:
        try:
            fn(df); print(f"[ok] {fn.__name__}")
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
    print("roc:", {k: v["auc"] for k, v in roc_data(df).items()})
    print("ALL OK")
