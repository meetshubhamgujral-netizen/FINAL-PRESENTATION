"""
BridgeCompliance Advisory - Market Validation Dashboard
=======================================================
A decision-first Streamlit dashboard that reads like a story: it opens with the
verdict, then walks through the evidence, and quarantines the technical proof in
a methodology appendix.

Tabs
  0  Executive Verdict            (Strategic Summary)
  1  Objectives & Approach        (Overview)
  2  Is the Market Real?          (Descriptive & Diagnostic)
  3  Who to Target                (Predictive)
  4  The Money                    (Prescriptive)
  5  The Playbook                 (Prescriptive)
  6  Methodology                  (Technical Appendix)

The dataset ships bundled - no upload needed.
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import analysis as A

# --------------------------------------------------------------------------- #
# Palette (validated, CVD-safe)                                               #
# --------------------------------------------------------------------------- #
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, MAGENTA, ORANGE = "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"
DUBAI, SING = BLUE, ORANGE
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ_BLUE = [[0, "#eaf2fd"], [0.5, "#5598e7"], [1, "#0d366b"]]
DIVERGING = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
TAG_COLORS = {"SUMMARY": ORANGE, "OVERVIEW": "#334155", "DESCRIPTIVE & DIAGNOSTIC": BLUE,
              "PREDICTIVE": VIOLET, "PRESCRIPTIVE": GREEN, "TECHNICAL APPENDIX": MUTED}

st.set_page_config(page_title="BridgeCompliance - Market Validation",
                   page_icon="B", layout="wide")

DATASET_NAME = "BridgeCompliance_Preprocessed_Dataset.xlsx"

# Currency (AED is pegged to USD at 3.6725; SGD legs are approximate mid-2026)
CUR = {"SGD": (1.0, "S$"), "AED": (2.79, "AED "), "USD": (0.76, "US$")}


def _find_dataset():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(here, "data", DATASET_NAME), os.path.join(here, DATASET_NAME)]:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(here, "**", "*.xlsx"), recursive=True)
    return sorted(hits)[0] if hits else os.path.join(here, DATASET_NAME)


DATA_PATH = _find_dataset()

# --------------------------------------------------------------------------- #
# CSS                                                                         #
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
.block-container {max-width: 1300px; padding-top: 1.1rem;}
h1,h2,h3,h4 {color:#0b0b0b;}
.tag{display:inline-block;font-size:10.5px;font-weight:700;color:#fff;padding:3px 12px;
     border-radius:20px;letter-spacing:.06em;margin:-6px 0 14px 0;}
.kpi{border:1px solid #e1e0d9;border-radius:12px;padding:12px 16px;background:#fff;height:100%;}
.kpi .lab{font-size:11px;color:#898781;text-transform:uppercase;letter-spacing:.05em;}
.kpi .val{font-size:23px;font-weight:700;color:#0b0b0b;line-height:1.15;}
.kpi .sub{font-size:11px;color:#898781;margin-top:2px;}
.kpi.b{border-left:4px solid #2a78d6;} .kpi.o{border-left:4px solid #eb6834;}
.kpi.g{border-left:4px solid #1baf7a;} .kpi.v{border-left:4px solid #4a3aa7;}
.callout{border-radius:10px;padding:14px 17px;margin:6px 0 15px 0;font-size:14px;
         line-height:1.6;color:#1b1b1b;}
.callout h4{margin:0 0 6px 0;font-size:13px;letter-spacing:.03em;text-transform:uppercase;}
.callout ul{margin:5px 0 0 0;padding-left:18px;} .callout li{margin:4px 0;}
.callout.story{background:#f6f8fc;border-left:4px solid #334155;}
.callout.why{background:#f4f2ff;border-left:4px solid #4a3aa7;}
.callout.take{background:#e9faf2;border-left:4px solid #1baf7a;}
.callout.note{background:#fff5e8;border-left:4px solid #eb6834;}
.card{border:1px solid #e1e0d9;border-radius:13px;padding:15px 17px;background:#fff;height:100%;}
.card .h{font-size:12px;color:#898781;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px;}
.card .a{font-size:15px;font-weight:700;color:#0b0b0b;line-height:1.35;}
.card p{font-size:13px;color:#3a3a38;margin:.3rem 0;line-height:1.5;}
.casc{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;}
.casc .step{flex:1;min-width:150px;border:1px solid #e1e0d9;border-radius:10px;padding:10px 12px;background:#fff;}
.casc .step .n{font-size:11px;color:#4a3aa7;font-weight:700;}
.casc .step .t{font-size:14px;font-weight:700;color:#0b0b0b;margin:2px 0;}
.casc .step .d{font-size:12px;color:#52514e;line-height:1.4;}
.funnel{max-width:780px;margin:4px auto 0 auto;}
.fstep{border:1px solid #e1e0d9;border-radius:12px;padding:13px 20px;text-align:center;}
.fstep .ft{font-weight:700;font-size:15.5px;color:#0b0b0b;}
.fstep .fs{font-size:13px;color:#52514e;margin-top:2px;line-height:1.4;}
.farr{text-align:center;color:#898781;font-size:15px;margin:4px 0;}
section[data-testid="stSidebar"]{background:#0f172a;}
section[data-testid="stSidebar"] *{color:#e2e8f0;}
.brand{font-size:18px;font-weight:800;color:#fff;} .brandsub{font-size:12px;color:#94a3b8;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data + cache                                                                #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_data(): return A.load_data(DATA_PATH)


if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found. Upload {DATASET_NAME} to the repo (root or data/).")
    st.stop()

df = get_data()


@st.cache_data(show_spinner="Training models on the full sample...")
def full(name, **kw):
    return getattr(A, name)(df, **kw)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def tag(label):
    st.markdown(f"<span class='tag' style='background:{TAG_COLORS[label]}'>{label}</span>",
                unsafe_allow_html=True)

def kpi(col, label, value, sub="", cls="b"):
    col.markdown(f"<div class='kpi {cls}'><div class='lab'>{label}</div>"
                 f"<div class='val'>{value}</div><div class='sub'>{sub}</div></div>",
                 unsafe_allow_html=True)

def callout(kind, title, body):
    head = f"<h4>{title}</h4>" if title else ""
    st.markdown(f"<div class='callout {kind}'>{head}{body}</div>", unsafe_allow_html=True)

def chart(fig, title, h=360, legend=True):
    st.markdown(f"<div style='font-weight:700;color:#0b0b0b;font-size:14.5px;"
                f"margin:2px 0 -4px 4px'>{title}</div>", unsafe_allow_html=True)
    st.plotly_chart(sty(fig, h, legend), width="stretch")

def sty(fig, h=360, legend=True):
    fig.update_layout(height=h, margin=dict(t=34 if legend else 12, b=10, l=8, r=8),
                      font=dict(family="system-ui,-apple-system,sans-serif", color=INK2, size=13),
                      paper_bgcolor="#fff", plot_bgcolor="#fff",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    if not legend:
        fig.update_layout(showlegend=False)
    return fig

def mkt_colors(vals):
    return {v: (DUBAI if "Dubai" in v else SING) for v in vals}


# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("<div class='brand'>BridgeCompliance</div>"
                "<div class='brandsub'>Market Validation Dashboard</div><br>",
                unsafe_allow_html=True)
    cur_name = st.selectbox("Currency", list(CUR.keys()), index=0)
    RATE, SYM = CUR[cur_name]
    st.caption("Rates are approximate. AED is pegged to USD at 3.6725; the SGD legs "
               "are indicative (mid-2026) - update before quoting a client.")
    st.markdown("**Filters**")
    markets = st.multiselect("Market", sorted(df["Q1_Market"].unique()),
                             default=sorted(df["Q1_Market"].unique()))
    stages = st.multiselect("Startup stage", A.STAGE_ORDER, default=A.STAGE_ORDER)
    inds = st.multiselect("Industry", sorted(df["Q2_Industry"].unique()),
                          default=sorted(df["Q2_Industry"].unique()))
    st.caption("Filters drive the Market and Playbook views. Predictive models train "
               "on the full sample (n=1,190) for stability.")


def money(sgd):
    return f"{SYM}{sgd * RATE:,.0f}"


dff = df[df["Q1_Market"].isin(markets) & df["Q3_Stage"].isin(stages)
         & df["Q2_Industry"].isin(inds)]
if dff.empty:
    st.warning("No rows match these filters - widen the selection in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# Header + tabs                                                               #
# --------------------------------------------------------------------------- #
st.title("BridgeCompliance Advisory - Market Validation Dashboard")
st.caption("Cross-border legal & compliance advisory - Dubai (DIFC/ADGM) vs "
           "Singapore (MAS) - synthetic dataset, n = 1,190 (pre-loaded)")

tabs = st.tabs([
    "0 - Executive Verdict", "1 - Objectives", "2 - Is the Market Real?",
    "3 - Who to Target", "4 - The Money", "5 - The Playbook",
    "6 - Recommendations", "7 - Methodology"])

# =========================================================================== #
# TAB 0 - EXECUTIVE VERDICT                                                    #
# =========================================================================== #
with tabs[0]:
    st.markdown("## Executive Verdict")
    tag("SUMMARY")

    cards = [
        ("Which market to enter", "Enter Dubai first - but the city is not the lever",
         "The two markets are statistically tied on demand - intent 45.6% (Dubai) vs "
         "44.3% (Singapore), and the significance tests find no real gap (p = 0.68 for "
         "intent, 0.97 for willingness to pay, 0.31 for urgency). However, Dubai skews "
         "to later-stage startups that pay more, so it has to be prioritised - yet the "
         "real advantage is the customer profile, not the location."),
        ("Who to target", "The 'Priority Converter' profile",
         "Seed to Series A-B, Fintech or Crypto, high urgency, 2-4 active regulatory "
         "pains. This group both wants the service and can pay for it. <i>Source: the "
         "K-Prototypes segmentation surfaced this as the top cluster (72% priority "
         "rate), and the classifier flagged stage and urgency as the strongest priority "
         "drivers.</i>"),
        ("How much they pay", "Price by maturity, not by city",
         "Willingness to pay rises cleanly with stage, from about SGD 2,000 to "
         "SGD 5,300 per month. Three tiers cover the market."),
        ("What to sell", "One bundle, not separate add-ons",
         "AML/CFT, Licensing and Data Protection are needed together far more often "
         "than chance - package them as a single offer."),
        ("The key risk", "Wanting is not the same as paying",
         "Interest and budget barely move together. The most enthusiastic lead is "
         "often not the one who pays, so we run two different sales plays."),
    ]
    r1 = st.columns(3)
    r2 = st.columns(3)
    for i, (h, a, p) in enumerate(cards):
        col = (r1 + r2)[i]
        col.markdown(f"<div class='card'><div class='h'>{h}</div>"
                     f"<div class='a'>{a}</div><p>{p}</p></div>", unsafe_allow_html=True)

    # base-case opportunity (scenario)
    shares = A.wtp_tier_shares(df)
    _, base_arr, base_clients = A.revenue_model(
        2500, 0.22, 0.05, {"Starter": 1500, "Growth": 3500, "Scale": 7000}, shares)
    r2[2].markdown(
        f"<div class='kpi g'><div class='lab'>Base-case opportunity (scenario)</div>"
        f"<div class='val'>{money(base_arr)}</div>"
        f"<div class='sub'>~{round(base_clients)} clients/yr - editable in The Money</div></div>",
        unsafe_allow_html=True)

    st.markdown("#### How we got to the answer")
    st.caption("Each step is powered by one technique - read it top to bottom to see "
               "how the data leads to the decision.")
    funnel = [
        ("Who's in the market?", "1,190 startups - Dubai and Singapore are tested and tied", "#eef1f6"),
        ("Who will engage?", "Classification - profile alone barely predicts (the honest finding)", "#eaf2fd"),
        ("Who's high-value?", "Segmentation - the 'Priority Converter' cluster rises to the top", "#eaf2fd"),
        ("Who can pay?", "Willingness to pay - the Growth and Scale tiers carry the revenue", "#f1eefb"),
        ("Who's ready now?", "Urgency and pain count - the priority profile", "#f1eefb"),
        ("What do we sell them?", "Association rules - the Core Compliance bundle", "#fff4e9"),
        ("The decision", "Target segment - price tier - dollar opportunity", "#e9faf2"),
    ]
    html = "<div class='funnel'>"
    for i, (t, s, bg) in enumerate(funnel):
        html += (f"<div class='fstep' style='background:{bg}'><div class='ft'>{t}</div>"
                 f"<div class='fs'>{s}</div></div>")
        if i < len(funnel) - 1:
            html += "<div class='farr'>&#8595;</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.write("")

    callout("take", "The one-line strategy",
            "<p>Do not bet on a city; bet on a profile. Enter through Dubai, lead with "
            "the compliance bundle for urgent Seed-to-Series-B fintechs, price by "
            "maturity, and spend the first sales hours on the highest expected-value "
            "leads. Everything else in this dashboard is the evidence for that sentence.</p>")

# =========================================================================== #
# TAB 1 - OBJECTIVES                                                           #
# =========================================================================== #
with tabs[1]:
    st.markdown("## Objectives & Approach")
    tag("OVERVIEW")
    callout("story", "",
            "<p><b>BridgeCompliance Advisory</b> is a cross-border legal & compliance "
            "consultancy for tech and fintech startups operating between <b>Dubai "
            "(DIFC/ADGM)</b> and <b>Singapore (MAS)</b>. The strategic goal is a "
            "<b>market entry</b> decision, so every analysis below is built to answer "
            "one of four commercial questions - not to be clever for its own sake.</p>")

    st.markdown("#### The four questions that drive the market-entry decision")
    qcols = st.columns(4)
    for col, h, t in [
        (qcols[0], "Target", "Which customer and market should we enter?"),
        (qcols[1], "Capacity", "How much can they pay?"),
        (qcols[2], "Behaviour", "How do they prefer to buy?"),
        (qcols[3], "Product", "Which service should we lead with?")]:
        col.markdown(f"<div class='card'><div class='h'>{h}</div>"
                     f"<p style='font-size:13.5px'>{t}</p></div>", unsafe_allow_html=True)

    st.markdown("#### The analytics ladder - and the objective of each rung")
    o = st.columns(3)
    o[0].markdown(
        "<div class='card'><div class='h' style='color:#2a78d6'>Descriptive & Diagnostic</div>"
        "<div class='a'>What is happening, and why</div>"
        "<p><b>Objective:</b> establish what the raw market actually looks like and "
        "test whether one city is genuinely a better bet, before committing resources.</p>"
        "<p><b>Output:</b> market comparison, significance tests, drivers of willingness "
        "to pay.</p></div>", unsafe_allow_html=True)
    o[1].markdown(
        "<div class='card'><div class='h' style='color:#4a3aa7'>Predictive</div>"
        "<div class='a'>Who is worth a sales call</div>"
        "<p><b>Objective:</b> from a startup's public profile alone, flag who will "
        "engage, who can pay, and who is a priority - so we spend scarce launch effort "
        "on the right leads.</p>"
        "<p><b>Output:</b> honest (leakage-free) scores and a targeting map.</p></div>",
        unsafe_allow_html=True)
    o[2].markdown(
        "<div class='card'><div class='h' style='color:#1baf7a'>Prescriptive</div>"
        "<div class='a'>What to do about it</div>"
        "<p><b>Objective:</b> turn the reading into an entry plan - segments, a revenue "
        "scenario, service bundles, pricing and a ranked call list.</p>"
        "<p><b>Output:</b> the money model and the go-to-market playbook.</p></div>",
        unsafe_allow_html=True)

    st.markdown("#### Two ideas to understand before reading the numbers")
    cc = st.columns(2)
    with cc[0]:
        callout("note", "The data is synthetic - and we are honest about it",
                "<p>The dataset was generated from estimated distributions, and the "
                "target labels (wants it, can pay, priority) were built from the survey "
                "answers themselves. So we always report performance <b>without leakage</b> "
                "- a lower but truthful number. A model that scores 100% here is copying "
                "the answer, not predicting it. Dollar figures in The Money are scenario "
                "math you control, not a forecast.</p>")
    with cc[1]:
        callout("why", "What a 'chain' of models means, in plain terms",
                "<p>Our three questions are not independent - they stack the way a sales "
                "person qualifies a lead: <b>interest first, budget second, priority "
                "last</b>. A startup only becomes a Priority if it wants the service "
                "<b>and</b> can afford it (in our data that holds ~99% of the time). A "
                "<b>Classifier Chain</b> predicts them in that order and passes each "
                "answer forward - 'wants it?' helps predict 'can pay?', and both help "
                "predict 'priority?'. Three separate models would pretend the questions "
                "are unrelated and waste that structure.</p>")

    k = st.columns(4)
    kpi(k[0], "Respondents", f"{len(df):,}", "after bot removal", "b")
    kpi(k[1], "Markets", "2", "Dubai - Singapore", "o")
    kpi(k[2], "Predictor features", f"{len(A.HONEST_FEATURES)}", "leakage-free set", "v")
    kpi(k[3], "Targets modelled", "3 + 1", "3 binary + WTP amount", "g")

# =========================================================================== #
# TAB 2 - IS THE MARKET REAL? (DESCRIPTIVE & DIAGNOSTIC)                       #
# =========================================================================== #
with tabs[2]:
    st.markdown("## Is the Market Real?")
    tag("DESCRIPTIVE & DIAGNOSTIC")
    callout("story", "",
            "<p>Before spending a dirham or a dollar on market entry, the first "
            "question is blunt: <b>is one city actually a better bet than the other?</b> "
            "We look at the raw demand signals side by side, then run formal tests to "
            "see whether any gap is real or just noise - and finally ask what truly "
            "moves willingness to pay.</p>")
    callout("why", "Why these methods (and not the usual ones)",
            "<ul>"
            "<li><b>Descriptive statistics first, no model.</b> You cannot model your "
            "way out of a market that isn't there. The right first step is to look.</li>"
            "<li><b>Spearman correlation, not Pearson.</b> Pearson assumes straight-line "
            "relationships between continuous, normally-distributed variables. Our data "
            "mixes 1-10 scales, yes/no flags and 3-tier prices - so Pearson's assumptions "
            "break. Spearman only asks whether things rise together, which is exactly the "
            "question and is valid on this data type.</li>"
            "<li><b>Significance tests to settle the market debate.</b> Chi-square (for a "
            "yes/no rate like intent), Mann-Whitney (for scores, without assuming a normal "
            "distribution) and Kruskal-Wallis (across the four stages) tell us whether a "
            "difference is real - so 'the cities are tied' is a tested statement, not an "
            "impression.</li></ul>")

    ms = A.market_summary(dff).reset_index()
    kcols = st.columns(5)
    kpi(kcols[0], "Respondents", f"{len(dff):,}", "current filter", "v")
    for i, (_, rr) in enumerate(ms.iterrows()):
        cls = "b" if "Dubai" in rr["Q1_Market"] else "o"
        short = "Dubai" if "Dubai" in rr["Q1_Market"] else "Singapore"
        kpi(kcols[1 + i], f"{short} intent", f"{rr['Intent_Rate']*100:.1f}%",
            f"avg WTP {money(rr['Avg_WTP_SGD'])}", cls)
    kpi(kcols[3], "Avg urgency", f"{dff['Q5_Urgency'].mean():.1f}/10", "all markets", "g")
    kpi(kcols[4], "High-priority", f"{dff['TARGET_Priority'].mean()*100:.0f}%", "of sample", "v")

    c1, c2 = st.columns(2)
    with c1:
        ws = A.wtp_by_stage(dff).reset_index(); ws.columns = ["Stage", "WTP"]
        fig = px.bar(ws, x="Stage", y="WTP", text="WTP", color_discrete_sequence=[BLUE])
        fig.update_traces(texttemplate="SGD %{text:,.0f}", textposition="outside", marker_line_width=0)
        fig.update_yaxes(title_text="WTP (SGD/mo)")
        chart(fig, "Willingness to pay climbs with startup stage", 360, False)
    with c2:
        ibs = A.intent_by_stage_market(dff)
        fig = px.bar(ibs, x="Q3_Stage", y="TARGET_High_Intent", color="Q1_Market",
                     barmode="group", color_discrete_map=mkt_colors(ibs["Q1_Market"].unique()))
        fig.update_yaxes(tickformat=".0%", title_text="Intent rate"); fig.update_xaxes(title_text="")
        chart(fig, "Intent rate by stage and market", 360, True)

    c3, c4 = st.columns(2)
    with c3:
        pp = A.pain_prevalence(dff)
        fig = px.bar(pp, x="Prevalence", y="Pain", color="Market", barmode="group",
                     orientation="h", color_discrete_map=mkt_colors(pp["Market"].unique()))
        fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(title_text="")
        chart(fig, "Regulatory pain prevalence, by market", 380, True)
    with c4:
        um = A.urgency_by_market(dff)
        fig = px.violin(um, x="Q1_Market", y="Q5_Urgency", color="Q1_Market", box=True,
                        color_discrete_map=mkt_colors(um["Q1_Market"].unique()))
        fig.update_xaxes(title_text=""); fig.update_yaxes(title_text="Urgency")
        chart(fig, "Urgency spreads look almost identical", 380, False)

    st.markdown("#### The test that settles it")
    st.dataframe(A.significance_tests(dff), width="stretch", hide_index=True)

    c5, c6 = st.columns([1.05, 1])
    with c5:
        sm = A.spearman_matrix(dff)
        fig = go.Figure(go.Heatmap(z=sm.values, x=sm.columns, y=sm.index, colorscale=DIVERGING,
                        zmid=0, zmin=-1, zmax=1, text=sm.values, texttemplate="%{text:.2f}",
                        textfont=dict(size=9)))
        chart(fig, "What moves together (Spearman correlation)", 430, False)
    with c6:
        sp = A.spearman_vs_wtp(dff)
        fig = px.bar(sp, x="rho", y="Driver", orientation="h", text="rho", color="rho",
                     color_continuous_scale=[[0, RED], [0.5, "#f0efec"], [1, BLUE]], range_color=[-0.6, 0.6])
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        fig.update_yaxes(title_text="", categoryorder="total ascending")
        chart(fig, "What drives willingness to pay", 430, False)

    callout("take", "What this means for market entry",
            "<p>The head-to-head looks like a tie, and the tests confirm it: the "
            "difference in intent, willingness to pay and urgency between Dubai and "
            "Singapore is <b>not statistically significant</b> (p-values well above "
            "0.05). What <b>is</b> highly significant is <b>startup stage</b> - a more "
            "mature startup reliably pays more, in either city. Strategic read: "
            "<b>choosing a market on the city name would be a mistake. The lever is the "
            "customer's maturity, so the entry plan must target a profile, not a "
            "location.</b></p>")

# =========================================================================== #
# TAB 3 - WHO TO TARGET (PREDICTIVE)                                           #
# =========================================================================== #
with tabs[3]:
    st.markdown("## Who to Target")
    tag("PREDICTIVE")
    callout("story", "",
            "<p>Entering a new market with a small team means we cannot chase everyone. "
            "So the practical question is: <b>from a startup's public profile alone - "
            "industry, stage, market, regulatory pains, urgency - can we tell who is "
            "worth a sales call before we make it?</b> That lets the launch team spend "
            "its first hours on the right doors.</p>")
    callout("why", "Why these models were chosen",
            "<ul>"
            "<li><b>Classification, because the decision is yes/no.</b> 'Will this "
            "startup engage?' is a category, not a quantity - so we classify.</li>"
            "<li><b>Tree models (Decision Tree, Random Forest, Gradient Boosting) over "
            "plain Logistic Regression.</b> Logistic assumes a smooth straight-line link "
            "between each variable and the outcome. Here the effects are more like "
            "thresholds - a startup isn't meaningfully keener at urgency 4 vs 5, but there "
            "may be a jump at 8. Trees capture those jumps and handle mixed variable types "
            "without distributional assumptions. We keep Logistic only as an interpretable "
            "baseline.</li>"
            "<li><b>A Classifier Chain, because the three answers are linked.</b> Priority "
            "= wants it AND can pay. The chain predicts in that order and feeds each answer "
            "into the next, mirroring the real qualification funnel; three separate models "
            "would throw that dependency away.</li>"
            "<li><b>We strip out the 'cheating' variables.</b> The intent label was built "
            "from a survey question, so leaving that question in lets the model copy the "
            "answer and score ~100%. We remove it and report the honest number - the one "
            "that survives contact with a real prospect.</li></ul>")

    leak = full("leakage_demonstration")
    l = st.columns(3)
    kpi(l[0], "With leakage", f"{leak['leaked_acc']*100:.0f}%", "model copies the answer", "o")
    kpi(l[1], "Honest (no leakage)", f"{leak['honest_acc']*100:.0f}%", "profile only", "b")
    kpi(l[2], "Pure chance", f"{leak['baseline']*100:.0f}%", "majority class", "v")
    st.caption("A 100% score is not skill - it is the model reading the answer it was "
               "given. The lower, honest number is what we trust to decide.")

    st.markdown("#### The targeting map: wanting is not the same as paying")
    qd, mx, my = A.quadrant_df(df)
    fig = px.scatter(qd, x="intent_jit", y="wtp_jit", color="Segment", opacity=0.5,
                     color_discrete_map={"Priority (wants & pays)": BLUE, "Other": "#c7c7c2"},
                     labels={"intent_jit": "Wants it  (likelihood to engage)",
                             "wtp_jit": "Can pay  (willingness to pay, SGD/mo)"})
    fig.add_vline(x=mx, line_dash="dash", line_color=MUTED)
    fig.add_hline(y=my, line_dash="dash", line_color=MUTED)
    ann = [(8.6, 6100, "Priority Converters<br>direct outreach", BLUE),
           (3.6, 6100, "Dormant whales<br>1:1 nurture", VIOLET),
           (8.6, 1250, "Fans on a budget<br>cheap self-serve", ORANGE),
           (3.6, 1250, "Deprioritise", MUTED)]
    for x, y, t, cclr in ann:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=11, color=cclr),
                           bgcolor="rgba(255,255,255,0.75)")
    chart(fig, "Intent vs willingness to pay - four plays, one per quadrant", 440, True)

    chain = full("fit_classifier_chain")
    c1, c2 = st.columns(2)
    with c1:
        pl = pd.DataFrame({"Link": list(chain["per_label_acc"].keys()),
                           "Acc": list(chain["per_label_acc"].values())})
        fig = px.bar(pl, x="Acc", y="Link", orientation="h", text="Acc", color_discrete_sequence=[VIOLET])
        fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        fig.update_xaxes(tickformat=".0%", range=[0, 1]); fig.update_yaxes(title_text="")
        chart(fig, "How well each link of the chain predicts", 320, False)
    with c2:
        fi = full("feature_importance_rf_gb").query("Model == 'Gradient Boosting'").head(8)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[BLUE])
        fig.update_yaxes(title_text="", categoryorder="total ascending")
        chart(fig, "What makes a startup a priority", 320, False)

    callout("take", "What this means for market entry",
            "<p>With only public-profile data we can pre-qualify leads meaningfully "
            "better than guessing - enough to sort the outreach list, not to decide "
            "blind. The map is the strategy: the <b>Priority Converters</b> (top-right) "
            "get direct sales effort; the <b>dormant whales</b> (can pay, not yet keen) "
            "get patient 1:1 nurture; the <b>fans on a budget</b> get a cheap self-serve "
            "product. And what most makes a startup a priority is its <b>stage and "
            "urgency</b> - which is exactly what a first sales question can uncover.</p>")

# =========================================================================== #
# TAB 4 - THE MONEY (PRESCRIPTIVE)                                             #
# =========================================================================== #
with tabs[4]:
    st.markdown("## The Money")
    tag("PRESCRIPTIVE")
    callout("story", "",
            "<p>A market-entry decision needs a number attached. But we will not pretend "
            "to forecast revenue from synthetic data - instead this is a <b>scenario "
            "model you drive</b>. The data supplies the <b>shape</b> (the mix of price "
            "tiers we actually see); you supply the <b>scale</b> (how big the market is "
            "and how well you convert). Move the sliders to pressure-test the opportunity.</p>")
    callout("why", "Why a scenario model, not a prediction",
            "<p>The dataset is synthetic and its labels were engineered, so any 'forecast' "
            "would be false precision. A transparent scenario is the honest and more useful "
            "tool: it makes every assumption explicit and lets a decision-maker see how the "
            "opportunity moves with market size, fit and conversion. The tier mix is taken "
            "from the real distribution in the data; the rest are levers you own.</p>")

    shares = A.wtp_tier_shares(df)
    st.markdown("#### Your assumptions")
    a1, a2, a3 = st.columns(3)
    tam_d = a1.number_input("Target startups - Dubai (DIFC/ADGM)", 100, 8000, 1000, 100,
                            help="DIFC passed 8,000 registered firms in 2025; its innovation "
                                 "hub hosts ~1,000+ fintech/innovation companies.")
    tam_s = a1.number_input("Target startups - Singapore (MAS)", 100, 8000, 1500, 100,
                            help="Singapore hosts roughly 1,500-1,600 fintech firms.")
    pct_fit = a2.slider("Share that fits the ideal profile (%)", 5, 60, 22, 1,
                        help="Our data suggests ~22% are Priority Converters.") / 100
    conv = a2.slider("Lead-to-client conversion (%)", 1, 30, 5, 1,
                     help="An assumption - measure it with a real pilot.") / 100
    months = a3.slider("Contract length (months)", 3, 24, 12, 1)
    p_start = a3.number_input("Starter price (SGD/mo)", 500, 4000, 1500, 100)
    p_grow = a3.number_input("Growth price (SGD/mo)", 1500, 8000, 3500, 100)
    p_scale = a3.number_input("Scale price (SGD/mo)", 3000, 15000, 7000, 100)

    dfm, arr, clients = A.revenue_model(tam_d + tam_s, pct_fit, conv,
                                        {"Starter": p_start, "Growth": p_grow, "Scale": p_scale},
                                        shares, months)
    k = st.columns(3)
    kpi(k[0], f"Modelled clients / yr", f"{round(clients)}", "reachable & converted", "b")
    kpi(k[1], f"Annual recurring revenue ({cur_name})", money(arr), f"over {months} months", "g")
    kpi(k[2], "Blended ticket / client", money(arr / clients if clients else 0),
        "per year", "v")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        show = dfm.copy()
        show["ARR"] = show["ARR (SGD)"].apply(money)
        show["Price"] = show["Price (SGD/mo)"].apply(money)
        st.dataframe(show[["Tier", "Clients", "Price", "ARR"]], width="stretch", hide_index=True)
    with c2:
        fig = px.bar(dfm, x="Tier", y="ARR (SGD)", text="ARR (SGD)",
                     color="Tier", color_discrete_sequence=[YELLOW, BLUE, GREEN])
        fig.update_traces(texttemplate="", textposition="outside")
        fig.update_yaxes(title_text=f"ARR (SGD)")
        chart(fig, "Where the revenue concentrates", 330, False)

    callout("take", "What this means for market entry",
            f"<p>At these assumptions the opportunity is about <b>{money(arr)} in annual "
            f"recurring revenue</b> from roughly <b>{round(clients)} clients</b> - and the "
            f"<b>Scale tier carries the most revenue despite being the smallest tier by "
            f"headcount</b>, which argues for investing early in the credibility that wins "
            f"later-stage clients. The lever that moves this number most is "
            f"<b>conversion</b>, and conversion is exactly what a small real-world pilot "
            f"would let us measure - the single highest-value next step before launch.</p>")

# =========================================================================== #
# TAB 5 - THE PLAYBOOK (PRESCRIPTIVE)                                          #
# =========================================================================== #
with tabs[5]:
    st.markdown("## The Playbook")
    tag("PRESCRIPTIVE")
    callout("story", "",
            "<p>This is where the reading becomes an entry plan: <b>which segments to "
            "serve, what to package, what to charge, whom to call first, and in what "
            "order to move.</b> Each piece answers a concrete launch question.</p>")
    callout("why", "Why these methods were chosen",
            "<ul>"
            "<li><b>K-Prototypes clustering, not K-Means.</b> K-Means measures "
            "straight-line distance, which is meaningless for categories - 'Fintech' is "
            "not numerically closer to 'Crypto' than to 'Payments'. K-Prototypes mixes "
            "numeric and categorical fields correctly, so the segments are real business "
            "types, not arithmetic artefacts.</li>"
            "<li><b>Apriori association rules for bundling.</b> The five regulatory pains "
            "are yes/no items - exactly the 'basket' structure Apriori was built for. It "
            "surfaces which pains co-occur more than chance (a lift above 1), which is the "
            "evidence for packaging services together.</li>"
            "<li><b>Expected-value ranking for the call list.</b> We sort leads by "
            "probability of being a priority times how much they would pay, so the team "
            "attacks the highest-value doors first, not merely the most likely.</li></ul>")

    prof = full("run_clustering")
    st.markdown("#### Customer segments - who clusters together")
    show = prof.drop(columns=["Stage"]).rename(
        columns={"StageLabel": "Stage", "WTP_SGD": "WTP", "Priority_Rate": "Priority %",
                 "Pain_Count": "Pains", "Dubai_%": "Dubai %"})
    show["Priority %"] = (show["Priority %"] * 100).round(0)
    show["WTP"] = show["WTP"].apply(money)
    st.dataframe(show[["Segment", "Profile", "Stage", "Urgency", "Likelihood", "WTP",
                       "Pains", "Priority %", "Dubai %", "N"]], width="stretch", hide_index=True)

    c1, c2 = st.columns([1.05, 1])
    with c1:
        pc = prof.copy()
        fig = px.scatter(pc, x="Urgency", y="WTP_SGD", size="N", color="Priority_Rate",
                         color_continuous_scale=SEQ_BLUE, size_max=55, text="Profile")
        fig.update_traces(textposition="top center", textfont=dict(size=11))
        fig.add_vline(x=pc["Urgency"].mean(), line_dash="dash", line_color=MUTED)
        fig.add_hline(y=pc["WTP_SGD"].mean(), line_dash="dash", line_color=MUTED)
        top = pc.loc[pc["Priority_Rate"].idxmax()]
        fig.add_annotation(x=top["Urgency"], y=top["WTP_SGD"] - 520, text="chase first",
                           showarrow=False, font=dict(size=11, color=GREEN))
        fig.update_layout(coloraxis_colorbar_title="Priority")
        fig.update_xaxes(title_text="Urgency"); fig.update_yaxes(title_text="WTP (SGD/mo)")
        chart(fig, "Priority matrix - urgency vs ability to pay", 400, False)
        st.caption("Top-right = urgent and high-paying = where sales effort goes first.")
    with c2:
        axes = ["Urgency", "Likelihood", "WTP_SGD", "Stage", "Pain_Count"]
        nm = prof.copy()
        for a in axes:
            nm[a] = (prof[a] - prof[a].min()) / (prof[a].max() - prof[a].min() + 1e-9)
        fig = go.Figure(); pal = [BLUE, AQUA, YELLOW, VIOLET]
        for i, (_, rr) in enumerate(nm.iterrows()):
            fig.add_trace(go.Scatterpolar(r=[rr[a] for a in axes] + [rr[axes[0]]],
                theta=["Urgency", "Likelihood", "WTP", "Stage", "Pains", "Urgency"],
                fill="toself", name=rr["Profile"], line_color=pal[i % 4]))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        chart(fig, "Segment fingerprints", 400, True)
        st.caption("The Priority Converter stretches furthest on every axis.")

    st.markdown("#### Service bundles - which compliance needs travel together")
    c3, c4 = st.columns([1, 1])
    with c3:
        sup = st.slider("Minimum support", 0.05, 0.40, 0.15, 0.05)
        lft = st.slider("Minimum lift", 1.0, 1.5, 1.0, 0.05)
        rules = full("association_rules_pain", min_support=sup, min_lift=lft)
        if rules.empty:
            st.warning("No rules at these thresholds - lower support or lift.")
        else:
            st.dataframe(rules, width="stretch", hide_index=True)
    with c4:
        rp = full("association_rules_pain", min_support=0.1, min_lift=1.0)
        if not rp.empty:
            fig = px.scatter(rp, x="Support", y="Confidence", size="Lift", color="Lift",
                             color_continuous_scale=SEQ_BLUE, size_max=28,
                             hover_data=["IF (has)", "THEN (also needs)"])
            chart(fig, "Rules: support vs confidence (bubble size = lift)", 360, False)
    st.caption("A lift above 1 means those pains appear together more than chance - the "
               "data-backed reason to sell them as one package, not separate add-ons.")

    st.markdown("#### Pricing tiers")
    pt = A.pricing_tiers()
    st.dataframe(pt, width="stretch", hide_index=True)

    st.markdown("#### Prioritised call list (highest expected value first)")
    leads = full("lead_ranking")
    leads = leads[leads["Market"].isin(markets) & leads["Stage"].isin(stages)]
    leads_disp = leads.copy()
    leads_disp["WTP (SGD/mo)"] = leads_disp["WTP (SGD/mo)"].apply(money)
    leads_disp["Expected_Value"] = leads["Expected_Value_SGD"].apply(money)
    st.dataframe(leads_disp[["ResponseID", "Market", "Industry", "Stage", "WTP (SGD/mo)",
                             "Urgency", "P_Priority", "Expected_Value"]],
                 width="stretch", hide_index=True)

    st.markdown("#### Go-to-market sequence")
    st.markdown(
        "<div class='casc'>"
        "<div class='step'><div class='n'>MONTH 1-2</div><div class='t'>Beachhead</div>"
        "<div class='d'>Dubai; urgent Seed-Series A-B fintech/crypto; bundle offer</div></div>"
        "<div class='step'><div class='n'>MONTH 3-4</div><div class='t'>Proof</div>"
        "<div class='d'>Convert priority leads; capture case studies and real conversion</div></div>"
        "<div class='step'><div class='n'>MONTH 5-6</div><div class='t'>Expand</div>"
        "<div class='d'>Open Singapore using Dubai proof; add Scale-tier retainers</div></div>"
        "<div class='step'><div class='n'>MONTH 6+</div><div class='t'>Systematise</div>"
        "<div class='d'>Self-serve for 'fans on a budget'; nurture dormant whales</div></div>"
        "</div>", unsafe_allow_html=True)

    b = prof.iloc[0]
    callout("take", "What this means for market entry",
            f"<p>The plan writes itself from the segments: pour direct effort into the "
            f"<b>{b['Profile']}</b> segment ({b['Priority_Rate']*100:.0f}% priority, top "
            f"willingness to pay), sell the <b>AML/CFT + Licensing + Data Protection "
            f"bundle</b> rather than loose add-ons, price by stage while using urgency as "
            f"the upsell trigger, and work the call list top-down by expected value. Enter "
            f"through Dubai, prove it, then carry the proof into Singapore.</p>")

# =========================================================================== #
# TAB 6 - RECOMMENDATIONS                                                      #
# =========================================================================== #
with tabs[6]:
    st.markdown("## Recommendations")
    tag("PRESCRIPTIVE")
    callout("story", "",
            "<p>This is the 'so what': the full analysis turned into a market-entry "
            "plan. Every recommendation is tied to a specific result from the tabs "
            "above, so the strategy is evidence-led, not opinion.</p>")

    recs = [
        ("1 - Launch market: enter Dubai first",
         "The descriptive read, the significance tests and the classifier all point "
         "the same way. Dubai and Singapore are statistically tied on demand, so the "
         "city is not the differentiator - but Dubai's base skews to later-stage, "
         "higher-paying startups, and stage is the single strongest driver of both "
         "willingness to pay and priority. Enter Dubai in months 1-6, build proof, "
         "then open Singapore in months 7-12 where MAS creates a more structured but "
         "slower buying process. <i>Evidence: Tab 2 (significance) + Tab 3 (feature "
         "importance).</i>"),
        ("2 - Ideal client profile: the Priority Converter",
         "Focus direct outreach on Seed to Series A-B fintech and crypto startups with "
         "high urgency and 2-4 active regulatory pains. This is the segment that both "
         "wants the service and can pay - it holds the highest willingness to pay and a "
         "72% priority rate. Everyone else is nurture, not outreach. <i>Evidence: Tab 5 "
         "(K-Prototypes segments) + Tab 3 (targeting map).</i>"),
        ("3 - Lead with the Core Compliance bundle",
         "Sell AML/CFT, Licensing and Data Protection as one package, not separate "
         "add-ons - startups that need one need the others far more often than chance. "
         "Frame it per market: in Dubai lead with VARA/DFSA licensing, AML programme "
         "design and token classification; in Singapore lead with the MAS licensing "
         "roadmap, PDPA cross-border data and AI governance. <i>Evidence: Tab 5 "
         "(association rules) + regulatory context.</i>"),
        ("4 - Price by maturity, upsell on urgency",
         "Use three tiers - Starter (Pre-seed/Seed), Growth (Series A-B), Scale "
         "(Series C+). Price scales with stage, but the trigger to move a client up a "
         "tier is urgency, not size: an urgent Pre-seed converts better than a lukewarm "
         "Series B. The Scale tier carries the most revenue despite the smallest "
         "headcount, so invest early in the credibility that wins later-stage clients. "
         "<i>Evidence: Tab 4 (revenue model) + Tab 2 (WTP by stage).</i>"),
        ("5 - Position on jurisdiction-specific expertise",
         "Make dual-jurisdiction, jurisdiction-specific expertise the brand anchor - it "
         "is the most-cited deal-breaker - and default to project-based engagements, "
         "offering annual retainers to Growth and Scale clients who want predictability. "
         "<i>Evidence: deal-breaker and engagement analysis in the dataset.</i>"),
    ]
    for h, b in recs:
        st.markdown(f"<div class='rec'><h4>{h}</h4><p>{b}</p></div>", unsafe_allow_html=True)

    callout("take", "The bottom line for market entry",
            f"<p>Enter through Dubai, target urgent Seed-to-Series-B fintechs with the "
            f"compliance bundle, price by maturity, and work the ranked lead list "
            f"top-down. At a conservative scenario that is roughly "
            f"{money(base_arr)} in annual recurring revenue - and the single highest-"
            f"value next step before committing budget is a small real-world pilot to "
            f"measure the true lead-to-contract conversion, the one number this "
            f"synthetic data cannot give us.</p>")

    st.markdown("#### What we would collect next time (to make this real)")
    st.markdown(
        "1. Measure the real outcome separately - did they sign, how much did they pay "
        "- instead of deriving it from the survey. This is what removes the leakage and "
        "turns the model from a pipeline demo into a real forecast.\n"
        "2. Run a 20-30 client pilot to learn the true intent-to-contract conversion.\n"
        "3. Add real firmographics (decision-maker role, company size, compliance budget).\n"
        "4. Run a conjoint pricing study with real service options and price points.\n"
        "5. Timestamp every record so the model can be validated against the future.")

# =========================================================================== #
# TAB 7 - METHODOLOGY (TECHNICAL APPENDIX)                                     #
# =========================================================================== #
with tabs[7]:
    st.markdown("## Methodology")
    tag("TECHNICAL APPENDIX")
    callout("story", "",
            "<p>The decision tabs stay clean on purpose; the full technical proof lives "
            "here. Every model, metric and honesty check is laid out so the work is "
            "auditable end to end.</p>")

    with st.expander("Classification - model comparison and how we score it", expanded=True):
        callout("why", "Why Recall is the priority metric",
                "<p>Missing a high-value client (a false negative) costs real revenue; "
                "flagging a low-intent startup by mistake (a false positive) costs one "
                "follow-up email. The costs are asymmetric, so we optimise for "
                "<b>Recall</b> and read <b>AUC</b>, not raw accuracy.</p>")
        st.dataframe(full("model_comparison"), width="stretch", hide_index=True)
        cc = st.columns(2)
        with cc[0]:
            roc = full("roc_data")
            fig = go.Figure()
            for i, (name, d) in enumerate(roc.items()):
                fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines",
                              name=f"{name} (AUC {d['auc']})",
                              line=dict(color=[BLUE, AQUA, YELLOW, VIOLET][i], width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                          line=dict(dash="dash", color=MUTED, width=1)))
            fig.update_xaxes(title_text="False positive rate"); fig.update_yaxes(title_text="True positive rate")
            chart(fig, "ROC curves (honest, leakage-free)", 380, True)
        with cc[1]:
            cms = full("confusion_matrices")
            key = list(cms.keys())[2]
            fig = px.imshow(cms[key], text_auto=True, color_continuous_scale=SEQ_BLUE,
                            labels=dict(x="Predicted", y="Actual", color="Cases"),
                            x=["No", "Yes"], y=["No", "Yes"])
            fig.update_coloraxes(showscale=False)
            chart(fig, f"Confusion matrix - {key}", 380, False)

    with st.expander("Cross-validation and the leakage demonstration"):
        cc = st.columns(2)
        with cc[0]:
            cvs = full("cv_scores")
            fig = px.bar(x=[f"Fold {i+1}" for i in range(len(cvs))], y=cvs,
                         text=[f"{v:.2f}" for v in cvs], color_discrete_sequence=[VIOLET])
            fig.update_traces(textposition="outside"); fig.update_layout(xaxis_title="", yaxis_title="F1")
            chart(fig, f"5-fold F1 stability (mean {cvs.mean():.2f})", 320, False)
        with cc[1]:
            st.dataframe(full("honest_classification_scores"), width="stretch", hide_index=True)
            leak = full("leakage_demonstration")
            st.caption(f"Leakage demonstration: adding the label's source question lifts "
                       f"accuracy from an honest {leak['honest_acc']*100:.0f}% to "
                       f"{leak['leaked_acc']*100:.0f}% - proof that the perfect score is "
                       f"copying, not predicting.")

    with st.expander("Regression - how much they pay (Ridge / Lasso)"):
        callout("why", "Why regularised regression",
                "<p>Urgency and likelihood move together (collinearity), which makes plain "
                "linear regression unstable. <b>Ridge</b> shrinks coefficients to stabilise "
                "them; <b>Lasso</b> pushes the weakest to zero, effectively selecting the "
                "real drivers. Predicting price on stage is legitimate signal, not leakage - "
                "stage is a firmographic, not derived from price.</p>")
        reg = full("wtp_regression")
        cc = st.columns([0.8, 1.1, 1.1])
        with cc[0]:
            mt = pd.DataFrame(reg["metrics"]).T.reset_index().rename(columns={"index": "Model"})
            st.dataframe(mt, width="stretch", hide_index=True)
        with cc[1]:
            cf = reg["coefs"].head(8).melt(id_vars="Feature", var_name="Model", value_name="Coef")
            fig = px.bar(cf, x="Coef", y="Feature", color="Model", barmode="group",
                         orientation="h", color_discrete_map={"Ridge": BLUE, "Lasso": ORANGE})
            fig.update_yaxes(title_text="", categoryorder="total ascending")
            chart(fig, "Ridge vs Lasso coefficients (SGD)", 340, True)
        with cc[2]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=reg["pred"], y=reg["resid"], mode="markers",
                          marker=dict(color=BLUE, opacity=0.35, size=6)))
            fig.add_hline(y=0, line_dash="dash", line_color=ORANGE)
            fig.update_xaxes(title_text="Predicted WTP (SGD)"); fig.update_yaxes(title_text="Residual")
            chart(fig, "Residual plot (honest caution)", 340, False)

    with st.expander("Clustering - choosing the number of segments"):
        callout("why", "Why the elbow method",
                "<p>Adding clusters always fits the data a little better, so the goal is "
                "not the lowest cost but the <b>elbow</b> - the point where extra segments "
                "stop meaningfully improving the fit. That lands at four, which is what we "
                "use.</p>")
        cc = A.cluster_cost_curve(df)
        fig = px.line(cc, x="k", y="cost", markers=True, color_discrete_sequence=[BLUE])
        fig.add_vline(x=4, line_dash="dash", line_color=ORANGE, annotation_text="k = 4")
        fig.update_xaxes(title_text="Number of segments (k)"); fig.update_yaxes(title_text="Cost")
        chart(fig, "K-Prototypes cost vs number of segments", 340, False)

    with st.expander("Data sufficiency and next-round collection plan"):
        st.dataframe(pd.DataFrame([
            {"Model": "Classification (chains)", "Enough?": "Yes (n)",
             "Caveat": "Labels engineered - measure real conversion next time"},
            {"Model": "Clustering (K-Prototypes)", "Enough?": "Yes", "Caveat": "Validate with silhouette"},
            {"Model": "Regression (Ridge/Lasso)", "Enough?": "Yes", "Caveat": "Only 3 WTP tiers - collect finer pricing"},
            {"Model": "Association (Apriori)", "Enough?": "Partial", "Caveat": "Only 5 binary pains - few rules"},
        ]), width="stretch", hide_index=True)
        st.markdown(
            "1. Measure the real outcome separately (did they sign, how much did they pay) "
            "instead of deriving it from the survey - this removes the leakage.\n"
            "2. Run a 20-30 client pilot to learn the true intent-to-contract conversion.\n"
            "3. Add real firmographics (decision-maker role, company size, compliance budget).\n"
            "4. Run a conjoint pricing study with real service options.\n"
            "5. Timestamp every record to validate against the future, not the past.")

    with st.expander("Keeping the model trustworthy in production (managerial view)"):
        st.markdown(
            "**Tuning, in plain terms.** Every model has knobs (how complex, how strict). "
            "Tuning is calibrating a scale: try combinations and keep the best - but "
            "calibrate on a *separate* slice from the one you measure on, or the good "
            "number is a lie. And tune to *not lose the valuable client* (a wrong 'no' "
            "costs a client; a wrong 'yes' costs an email), not for generic accuracy.\n\n"
            "**Champion vs challenger.** Always compare a new model against the one in use. "
            "The challenger only goes live if it wins on real data, and first on a small "
            "group; if it fails, roll back in minutes.")
        st.dataframe(pd.DataFrame([
            {"What to watch": "Did the type of client change?", "What to do": "Check monthly; retrain if it shifts"},
            {"What to watch": "Is accuracy dropping vs real outcomes?", "What to do": "Retrain - regulation moves"},
            {"What to watch": "Do the '70% sure' really hit 70%?", "What to do": "Recalibrate probabilities"},
            {"What to watch": "Borderline (grey-zone) cases", "What to do": "A person reviews, not the model alone"},
        ]), width="stretch", hide_index=True)

st.divider()
st.caption("BridgeCompliance Advisory - Angie Ximena Lozano Rincon - "
           "Data Analytics Term 2 - SP Jain Global School of Management")
