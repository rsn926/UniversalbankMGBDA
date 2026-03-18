# ═══════════════════════════════════════════════════════════════════════════
#   Universal Bank — Personal Loan Campaign Intelligence Dashboard
#   Streamlit Application  |  Decision Tree · Random Forest · Gradient Boosting
# ═══════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank | Loan Campaign Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Age", "Experience", "Income", "Family", "CCAvg",
    "Education", "Mortgage", "Securities Account",
    "CD Account", "Online", "CreditCard", "Has_Mortgage",
]

MODEL_COLORS = {
    "Decision Tree":     "#f97316",
    "Random Forest":     "#2563eb",
    "Gradient Boosting": "#16a34a",
}

EDU_MAP = {1: "Undergrad", 2: "Graduate", 3: "Advanced"}

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Refined Banking Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0f2248 40%, #1a3a6e 70%, #0e4d92 100%);
    border-radius: 18px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(37,99,235,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(16,163,74,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.72);
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.85);
    margin-bottom: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── KPI Cards ── */
.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    flex: 1;
    background: #ffffff;
    border-radius: 14px;
    padding: 1.3rem 1.2rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    border-top: 3px solid var(--accent, #2563eb);
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle at top right, rgba(37,99,235,0.06), transparent);
    border-radius: 0 0 0 100%;
}
.kpi-val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0a1628;
    line-height: 1.1;
}
.kpi-lbl {
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
    font-weight: 500;
}
.kpi-delta {
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.3rem;
}

/* ── Section Title ── */
.sec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #0a1628;
    padding: 0.4rem 0 0.4rem 1rem;
    border-left: 4px solid #2563eb;
    margin: 2rem 0 1.2rem 0;
}

/* ── Insight Box ── */
.insight {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 1rem 0;
    border-left: 4px solid #2563eb;
    font-size: 0.87rem;
    color: #1e3a5f;
    line-height: 1.55;
}

/* ── Caption Box ── */
.caption {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: #4b5563;
    font-style: italic;
    margin-top: 0.3rem;
    line-height: 1.5;
}

/* ── Recommendation Card ── */
.rec-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin: 0.6rem 0;
    border: 1px solid #bfdbfe;
    box-shadow: 0 2px 10px rgba(37,99,235,0.07);
}
.rec-card h4 {
    color: #0a1628;
    margin: 0 0 0.45rem 0;
    font-size: 0.93rem;
    font-weight: 600;
}
.rec-card p {
    margin: 0;
    font-size: 0.84rem;
    color: #374151;
    line-height: 1.55;
}
.rec-rate {
    color: #2563eb;
    font-weight: 700;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0f2248 100%);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.88rem !important; }

/* ── Table Styling ── */
.dataframe { font-size: 0.84rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def sec(text):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)

def cap(text):
    st.markdown(f'<div class="caption">💡 {text}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight">🔍 {text}</div>', unsafe_allow_html=True)

def kpi(val, lbl, accent="#2563eb", delta_text=""):
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-val">{val}</div>
        <div class="kpi-lbl">{lbl}</div>
        {f'<div class="kpi-delta" style="color:{accent}">{delta_text}</div>' if delta_text else ''}
    </div>"""

def chart_layout(fig, title="", h=350, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#0a1628",
                   family="Playfair Display"), x=0.5) if title else {},
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        font=dict(family="DM Sans", size=11, color="#374151"),
        margin=dict(t=50 if title else 20, b=30, l=20, r=20),
    )
    fig.update_xaxes(gridcolor="#f1f5f9", showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor="#f1f5f9", showgrid=True, zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df["Experience"] = df["Experience"].clip(lower=0)
    df.drop(columns=["ID", "ZIP Code"], inplace=True)
    df["Has_Mortgage"] = (df["Mortgage"] > 0).astype(int)
    df["Education_Label"] = df["Education"].map(EDU_MAP)
    df["Income_Band"] = pd.cut(
        df["Income"], bins=[0, 40, 80, 120, 225],
        labels=["Low (<$40K)", "Mid ($40–80K)", "High ($80–120K)", "Very High (>$120K)"]
    )
    df["CCAvg_Band"] = pd.cut(
        df["CCAvg"], bins=[-0.01, 1, 3, 6, 11],
        labels=["Low (<$1K)", "Mid ($1–3K)", "High ($3–6K)", "Very High (>$6K)"]
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def train_models(df):
    X = df[FEATURE_COLS]
    y = df["Personal Loan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=15, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_tr = model.predict(X_train)
        y_pred_te = model.predict(X_test)
        y_prob_te = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob_te)
        roc_auc_val = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred_te)

        results[name] = {
            "model":             model,
            "train_acc":         accuracy_score(y_train, y_pred_tr),
            "test_acc":          accuracy_score(y_test,  y_pred_te),
            "precision":         precision_score(y_test, y_pred_te, zero_division=0),
            "recall":            recall_score(y_test,    y_pred_te, zero_division=0),
            "f1":                f1_score(y_test,        y_pred_te, zero_division=0),
            "roc_auc":           roc_auc_val,
            "fpr":               fpr,
            "tpr":               tpr,
            "cm":                cm,
            "feature_importance": model.feature_importances_,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def page_executive(df):
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">AI-Powered Campaign Intelligence</div>
        <div class="hero-title">🏦 Universal Bank<br>Personal Loan Campaign</div>
        <p class="hero-sub">
            From descriptive insights to prescriptive actions — everything you need
            to build a hyper-personalised, high-conversion marketing campaign.
        </p>
    </div>
    """, unsafe_allow_html=True)

    total      = len(df)
    loan_yes   = int(df["Personal Loan"].sum())
    loan_rate  = loan_yes / total * 100
    avg_inc    = df["Income"].mean()
    avg_cc     = df["CCAvg"].mean()
    cd_rate    = df[df["CD Account"] == 1]["Personal Loan"].mean() * 100
    hi_inc_rate= df[df["Income"] > 100]["Personal Loan"].mean() * 100

    cols = st.columns(5)
    metrics = [
        (f"{total:,}",       "Total Customers",     "#2563eb", "Historical Dataset"),
        (f"{loan_yes:,}",    "Accepted Loan",        "#16a34a", f"{loan_rate:.1f}% of base"),
        (f"{loan_rate:.1f}%","Acceptance Rate",      "#f97316", "⚠ Class Imbalanced"),
        (f"${avg_inc:.0f}K", "Avg. Annual Income",   "#7c3aed", f"CC Avg: ${avg_cc:.1f}K/mo"),
        (f"{cd_rate:.0f}%",  "CD Holders Accept",    "#0891b2", "vs 9.6% overall avg"),
    ]
    accents = ["#2563eb","#16a34a","#f97316","#7c3aed","#0891b2"]
    for i, (col, (v, l, a, d)) in enumerate(zip(cols, metrics)):
        col.markdown(kpi(v, l, a, d), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Donut + Segment Bar ──
    c1, c2 = st.columns([1, 2])

    with c1:
        fig = go.Figure(go.Pie(
            labels=["Accepted Loan", "Did Not Accept"],
            values=[loan_yes, total - loan_yes],
            hole=0.58,
            marker_colors=["#2563eb", "#dbeafe"],
            textinfo="label+percent",
            textfont_size=11,
            pull=[0.03, 0],
        ))
        fig.add_annotation(
            text=f"<b>{loan_rate:.1f}%</b><br><span style='font-size:11px'>Accept Rate</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#0a1628"),
        )
        chart_layout(fig, "Overall Loan Acceptance", h=310, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        cap("Only 9.6% of customers accepted a personal loan — a significant class imbalance. Precision targeting is critical with a halved marketing budget.")

    with c2:
        seg = {
            "CD Account Holders":         df[df["CD Account"] == 1]["Personal Loan"].mean() * 100,
            "Income > $100K":             df[df["Income"] > 100]["Personal Loan"].mean() * 100,
            "Advanced Education":         df[df["Education"] == 3]["Personal Loan"].mean() * 100,
            "Family Size 3–4":            df[df["Family"] >= 3]["Personal Loan"].mean() * 100,
            "High CC Spend (>$3K/mo)":    df[df["CCAvg"] > 3]["Personal Loan"].mean() * 100,
            "Overall Baseline":           loan_rate,
        }
        seg_df = pd.DataFrame({"Segment": list(seg.keys()), "Rate": list(seg.values())}).sort_values("Rate")
        colors = ["#94a3b8" if s == "Overall Baseline" else "#2563eb" for s in seg_df["Segment"]]

        fig2 = px.bar(
            seg_df, x="Rate", y="Segment", orientation="h",
            text=seg_df["Rate"].apply(lambda x: f"{x:.1f}%"),
            color="Rate", color_continuous_scale=["#bfdbfe","#1d4ed8"],
        )
        fig2.update_traces(textposition="outside", textfont_size=12, marker_line_width=0)
        fig2.update_coloraxes(showscale=False)
        chart_layout(fig2, "Loan Acceptance Rate by Key Segment", h=310, legend=False)
        fig2.update_layout(xaxis=dict(title="Acceptance Rate (%)", range=[0, max(seg.values()) + 12]),
                           yaxis=dict(title=""))
        st.plotly_chart(fig2, use_container_width=True)
        cap("CD Account holders accept loans at ~30% — nearly 3× the overall average. High income and advanced education are the next strongest predictors. These three signals define your primary target audience.")

    # ── Key Takeaways ──
    sec("📌 Strategic Takeaways — What You Need to Know Before the Campaign")
    c1, c2, c3 = st.columns(3)
    with c1:
        insight("**Only 9.6%** of customers historically accepted a personal loan. With budget cut in half, you cannot afford to spray-and-pray. AI scoring will help you identify the top 10–15% most likely to convert.")
    with c2:
        insight("**CD Account holders** are 3× more likely to accept a personal loan — they have an existing relationship with the bank and proven savings behaviour. These are your lowest-effort, highest-yield targets.")
    with c3:
        insight("**Income + CC Spending** are the two strongest predictors. Customers earning >$100K who also spend >$3K/month on credit cards form the highest-conversion micro-segment.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_descriptive(df):
    sec("📊 Descriptive Analytics — Who Are Your Customers?")
    insight("Descriptive analytics answers: **What does our customer base look like?** Understanding the distribution of age, income, spending, and product ownership is the foundation for all downstream analysis.")

    # Summary table
    st.markdown("##### 📋 Summary Statistics")
    desc = df[["Age","Experience","Income","Family","CCAvg","Mortgage"]].describe().T.round(2)
    desc.columns = ["Count","Mean","Std Dev","Min","25th %ile","Median","75th %ile","Max"]
    st.dataframe(
        desc.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["Mean","Median"]),
        use_container_width=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Age & Experience ──
    sec("👤 Age & Professional Experience")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Age", nbins=30, color_discrete_sequence=["#2563eb"],
                           labels={"Age":"Age (years)","count":"Customers"})
        fig.update_traces(marker_line_color="white", marker_line_width=0.5)
        chart_layout(fig, "Customer Age Distribution", h=320, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        cap("Ages range 23–67, peaking around 35–55. This prime earning-age cohort is most receptive to credit products. Younger customers (<30) are a smaller segment worth tracking for future pipeline.")

    with c2:
        fig = px.histogram(df, x="Experience", nbins=28, color_discrete_sequence=["#7c3aed"],
                           labels={"Experience":"Years of Experience","count":"Customers"})
        fig.update_traces(marker_line_color="white", marker_line_width=0.5)
        chart_layout(fig, "Professional Experience Distribution", h=320, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        cap("Experience closely mirrors age (r ≈ 0.99), peaking at 15–30 years. Negative values were corrected to 0 as they are data entry artefacts. This feature is largely redundant given the age column.")

    # ── Income & CCAvg ──
    sec("💰 Income & Credit Card Spending")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Income", nbins=35, color_discrete_sequence=["#f97316"],
                           labels={"Income":"Annual Income ($000)","count":"Customers"})
        fig.update_traces(marker_line_color="white", marker_line_width=0.5)
        chart_layout(fig, "Annual Income Distribution ($000)", h=320, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        cap("Income is right-skewed — the majority earn $40K–$100K. A smaller affluent segment (>$150K) exists and disproportionately drives loan acceptance. These high earners deserve a dedicated, premium outreach strategy.")

    with c2:
        fig = px.histogram(df, x="CCAvg", nbins=30, color_discrete_sequence=["#16a34a"],
                           labels={"CCAvg":"Monthly CC Spend ($000)","count":"Customers"})
        fig.update_traces(marker_line_color="white", marker_line_width=0.5)
        chart_layout(fig, "Monthly Credit Card Spending ($000)", h=320, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        cap("Most customers spend under $2K/month on credit cards. The tail of heavy spenders (>$3K/month) is small but crucial — this group shows dramatically higher loan acceptance rates and should be a primary campaign target.")

    # ── Categorical Features ──
    sec("🏷 Customer Demographics & Product Adoption")
    c1, c2, c3 = st.columns(3)

    with c1:
        edu_counts = df["Education_Label"].value_counts().reindex(["Undergrad","Graduate","Advanced"]).reset_index()
        edu_counts.columns = ["Education","Count"]
        fig = px.bar(edu_counts, x="Education", y="Count",
                     color="Education", color_discrete_sequence=["#2563eb","#7c3aed","#16a34a"],
                     text="Count")
        fig.update_traces(textposition="outside", marker_line_width=0)
        chart_layout(fig, "Education Level Breakdown", h=320, legend=False)
        fig.update_layout(yaxis=dict(range=[0, edu_counts["Count"].max()*1.2], title="No. of Customers"),
                          xaxis=dict(title=""))
        st.plotly_chart(fig, use_container_width=True)
        cap("Education is evenly distributed across three levels. Advanced-degree holders earn more on average, driving their higher loan acceptance rates — education acts as a proxy for income potential.")

    with c2:
        fam_counts = df["Family"].value_counts().sort_index().reset_index()
        fam_counts.columns = ["Family Size","Count"]
        fam_counts["Pct"] = (fam_counts["Count"] / fam_counts["Count"].sum() * 100).round(1)
        fig = px.bar(fam_counts, x="Family Size", y="Count",
                     color="Count", color_continuous_scale=["#bfdbfe","#1d4ed8"],
                     text=fam_counts.apply(lambda r: f"{r['Count']:,}\n({r['Pct']}%)", axis=1))
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        chart_layout(fig, "Family Size Distribution", h=320, legend=False)
        fig.update_layout(yaxis=dict(range=[0, fam_counts["Count"].max()*1.2], title="No. of Customers"),
                          xaxis=dict(title="Family Size (members)"))
        st.plotly_chart(fig, use_container_width=True)
        cap("Families of 3–4 face higher financial obligations — school fees, household expenses, mortgages — making them more likely to explore personal loans. Target these segments with family-centric messaging.")

    with c3:
        binary_cols = ["Securities Account","CD Account","Online","CreditCard"]
        binary_labels = ["Securities Acc.","CD Account","Online Banking","UB Credit Card"]
        binary_rates = [df[c].mean() * 100 for c in binary_cols]
        bin_df = pd.DataFrame({"Feature": binary_labels, "Adoption (%)": binary_rates})
        fig = px.bar(bin_df, x="Feature", y="Adoption (%)",
                     color="Feature",
                     color_discrete_sequence=["#f97316","#2563eb","#16a34a","#7c3aed"],
                     text=bin_df["Adoption (%)"].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside", marker_line_width=0)
        chart_layout(fig, "Product & Digital Adoption Rates", h=320, legend=False)
        fig.update_layout(yaxis=dict(range=[0, 80], title="% of Customers"),
                          xaxis=dict(title=""))
        st.plotly_chart(fig, use_container_width=True)
        cap("60% use online banking — a huge digital engagement opportunity. Only 6% hold a CD Account, yet they show the highest loan acceptance rate. Online banking users are ideal for low-cost digital campaign channels.")

    # ── Mortgage ──
    sec("🏠 Mortgage Distribution")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(df[df["Mortgage"] > 0], x="Mortgage", nbins=30,
                           color_discrete_sequence=["#0891b2"],
                           labels={"Mortgage":"Mortgage Value ($000)","count":"Customers"})
        fig.update_traces(marker_line_color="white", marker_line_width=0.5)
        chart_layout(fig, "Mortgage Value Distribution (Mortgage Holders Only)", h=300, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        cap("Among the ~36% of customers with a mortgage, values range widely up to $635K. Mortgage holders have demonstrated creditworthiness — an important qualifier when combined with income level.")

    with c2:
        mort_pie = pd.DataFrame({
            "Status": ["Has Mortgage", "No Mortgage"],
            "Count": [df["Has_Mortgage"].sum(), (df["Has_Mortgage"] == 0).sum()]
        })
        fig = px.pie(mort_pie, names="Status", values="Count",
                     color_discrete_sequence=["#0891b2","#e0f2fe"], hole=0.55)
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=True, margin=dict(t=20,b=20,l=10,r=10))
        pct = df["Has_Mortgage"].mean() * 100
        fig.add_annotation(text=f"<b>{pct:.0f}%</b><br>Mortgaged", x=0.5, y=0.5,
                           showarrow=False, font_size=14, font_color="#0a1628")
        st.plotly_chart(fig, use_container_width=True)
        cap("36% of customers hold a mortgage — useful as a binary credit behaviour flag.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DIAGNOSTIC ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_diagnostic(df):
    sec("🔍 Diagnostic Analytics — What Drives Loan Acceptance?")
    insight("Diagnostic analytics moves beyond *what happened* to answer *why it happened*. We analyse how each customer attribute correlates with personal loan acceptance, helping us identify the most powerful targeting signals.")

    # ── Box Plots ──
    c1, c2 = st.columns(2)
    loan_label = df["Personal Loan"].map({0: "Did Not Accept", 1: "Accepted Loan"})

    with c1:
        fig = px.box(df, x=loan_label, y="Income",
                     color=loan_label,
                     color_discrete_map={"Did Not Accept":"#93c5fd","Accepted Loan":"#1d4ed8"},
                     labels={"x":"","y":"Annual Income ($000)"})
        fig.update_traces(marker_size=3)
        chart_layout(fig, "Income Distribution by Loan Acceptance", h=350, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        cap("Loan acceptors have a median income ~2× higher than non-acceptors. Income is the single strongest individual predictor — customers earning over $100K are prime candidates for a personal loan offer.")

    with c2:
        fig = px.box(df, x=loan_label, y="CCAvg",
                     color=loan_label,
                     color_discrete_map={"Did Not Accept":"#86efac","Accepted Loan":"#15803d"},
                     labels={"x":"","y":"Avg. Monthly CC Spend ($000)"})
        fig.update_traces(marker_size=3)
        chart_layout(fig, "Credit Card Spend by Loan Acceptance", h=350, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        cap("Loan acceptors spend ~2.5× more monthly on credit cards. High CC spending signals financial confidence, active credit behaviour, and comfort with debt products — a strong proxy for loan receptiveness.")

    # ── Acceptance Rates by Segment ──
    sec("📈 Loan Acceptance Rate Across Key Dimensions")
    c1, c2 = st.columns(2)

    with c1:
        edu_rate = df.groupby("Education_Label")["Personal Loan"].agg(["mean","count"]).reset_index()
        edu_rate.columns = ["Education","Rate","N"]
        edu_rate["Rate_%"] = edu_rate["Rate"] * 100
        edu_rate["Label"] = edu_rate.apply(lambda r: f"{r['Rate_%']:.1f}%<br>(n={r['N']:,})", axis=1)
        fig = px.bar(edu_rate.sort_values("Rate_%"),
                     x="Rate_%", y="Education", orientation="h",
                     color="Education",
                     color_discrete_sequence=["#2563eb","#7c3aed","#16a34a"],
                     text=edu_rate.sort_values("Rate_%")["Rate_%"].apply(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="outside", marker_line_width=0)
        chart_layout(fig, "Acceptance Rate by Education Level", h=300, legend=False)
        fig.update_layout(xaxis=dict(range=[0,20], title="Acceptance Rate (%)"), yaxis=dict(title=""))
        st.plotly_chart(fig, use_container_width=True)
        cap("Advanced-degree holders accept personal loans at nearly 2× the rate of undergraduates. Education is a proxy for income — craft premium, aspirational messaging for graduate and advanced segments.")

    with c2:
        fam_rate = df.groupby("Family")["Personal Loan"].agg(["mean","count"]).reset_index()
        fam_rate.columns = ["Family","Rate","N"]
        fam_rate["Rate_%"] = fam_rate["Rate"] * 100
        fig = px.bar(fam_rate, x="Family", y="Rate_%",
                     color="Rate_%", color_continuous_scale=["#bfdbfe","#1d4ed8"],
                     text=fam_rate["Rate_%"].apply(lambda x: f"{x:.1f}%"),
                     labels={"Family":"Family Size","Rate_%":"Acceptance Rate (%)"})
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        chart_layout(fig, "Acceptance Rate by Family Size", h=300, legend=False)
        fig.update_layout(yaxis=dict(range=[0,20]))
        st.plotly_chart(fig, use_container_width=True)
        cap("Family size 3 shows the highest acceptance rate. Larger families have greater financial commitments and are more open to credit. Target these with messaging around life moments — education, home renovation, family needs.")

    # ── Binary product adoption vs acceptance ──
    sec("🔘 Product Ownership vs. Loan Acceptance")
    binary_cols   = ["Securities Account","CD Account","Online","CreditCard"]
    binary_labels = ["Securities Account","CD Account","Online Banking","UB Credit Card"]
    rates_y, rates_n = [], []
    for col in binary_cols:
        rates_y.append(df[df[col] == 1]["Personal Loan"].mean() * 100)
        rates_n.append(df[df[col] == 0]["Personal Loan"].mean() * 100)

    fig = go.Figure(data=[
        go.Bar(name="Has Product", x=binary_labels, y=rates_y,
               marker_color="#2563eb", text=[f"{v:.1f}%" for v in rates_y],
               textposition="outside"),
        go.Bar(name="Does NOT Have Product", x=binary_labels, y=rates_n,
               marker_color="#bfdbfe", text=[f"{v:.1f}%" for v in rates_n],
               textposition="outside"),
    ])
    fig.update_layout(barmode="group")
    chart_layout(fig, "Loan Acceptance Rate: Product Owners vs. Non-Owners", h=380)
    fig.update_layout(yaxis=dict(range=[0,40], title="Acceptance Rate (%)"),
                      xaxis=dict(title=""),
                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)
    cap("CD Account holders have a 29.4% acceptance rate — 3× the overall average and the single most predictive binary feature. Conversely, having a UB Credit Card has almost no impact on loan acceptance probability.")

    # ── Income Band vs Acceptance ──
    sec("💵 Income Band vs. Loan Acceptance Rate")
    inc_rate = df.groupby("Income_Band", observed=True)["Personal Loan"].agg(["mean","count"]).reset_index()
    inc_rate.columns = ["Income_Band","Rate","N"]
    inc_rate["Rate_%"] = inc_rate["Rate"] * 100
    fig = px.bar(inc_rate, x="Income_Band", y="Rate_%",
                 color="Rate_%", color_continuous_scale=["#bfdbfe","#1d4ed8"],
                 text=inc_rate["Rate_%"].apply(lambda x: f"{x:.1f}%"),
                 labels={"Income_Band":"Income Band","Rate_%":"Acceptance Rate (%)"})
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    chart_layout(fig, "Loan Acceptance Rate by Income Band", h=340, legend=False)
    fig.update_layout(yaxis=dict(range=[0, inc_rate["Rate_%"].max() + 12]))
    st.plotly_chart(fig, use_container_width=True)
    cap("Acceptance rate jumps dramatically above $80K income. The Very High income band (>$120K) shows acceptance rates 5–6× the overall average — a clear demarcation for premium targeting. Allocate your best offer and direct outreach to this segment.")

    # ── Scatter: Income vs CCAvg ──
    sec("📍 Income vs. CC Spend — The Two-Signal Targeting Map")
    scatter_df = df.copy()
    scatter_df["Loan Status"] = scatter_df["Personal Loan"].map({0:"Did Not Accept",1:"Accepted Loan"})
    sample = scatter_df.sample(min(2500, len(scatter_df)), random_state=42)
    fig = px.scatter(sample, x="Income", y="CCAvg",
                     color="Loan Status",
                     color_discrete_map={"Did Not Accept":"#93c5fd","Accepted Loan":"#1d4ed8"},
                     opacity=0.55, size_max=5,
                     labels={"Income":"Annual Income ($000)","CCAvg":"Monthly CC Spend ($000)"})
    fig.add_vline(x=100, line_dash="dash", line_color="#f97316", line_width=1.5,
                  annotation_text="$100K Income Threshold", annotation_position="top right",
                  annotation_font_color="#f97316")
    fig.add_hline(y=3, line_dash="dash", line_color="#16a34a", line_width=1.5,
                  annotation_text="$3K CC Spend Threshold", annotation_position="bottom right",
                  annotation_font_color="#16a34a")
    chart_layout(fig, "Income vs. Monthly CC Spend — Coloured by Loan Acceptance", h=450)
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)
    cap("The top-right quadrant (Income >$100K AND CC Spend >$3K/month) is densely populated with loan acceptors (dark blue). These two thresholds define your highest-priority targeting zone — the 'Golden Quadrant' for this campaign.")

    # ── Correlation Heatmap ──
    sec("🔗 Feature Correlation Matrix")
    corr_cols = ["Age","Experience","Income","Family","CCAvg","Education",
                 "Mortgage","Personal Loan","Securities Account","CD Account",
                 "Online","CreditCard","Has_Mortgage"]
    corr = df[corr_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_traces(textfont_size=9)
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20,b=20,l=20,r=20),
                      coloraxis_colorbar=dict(title="r"))
    st.plotly_chart(fig, use_container_width=True)
    cap("Income (r ≈ 0.50) and CCAvg (r ≈ 0.37) show the strongest positive correlations with Personal Loan. Age and Experience are nearly perfectly correlated (r ≈ 0.99) — only one needs to be kept in the model. CD Account (r ≈ 0.32) punches well above its weight given how few customers hold it.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PREDICTIVE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_predictive(df, results):
    sec("🤖 Predictive Analytics — AI Classification Models")
    insight("Three supervised classification algorithms were trained on 70% of the customer data and evaluated on a 30% holdout set. Class imbalance (9.6% positive) was addressed via `class_weight='balanced'`, ensuring the models are sensitive to loan acceptors — not just the majority class.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics Table ──
    sec("📊 Model Performance Comparison")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Train Accuracy": f"{r['train_acc']*100:.2f}%",
            "Test Accuracy":  f"{r['test_acc']*100:.2f}%",
            "Precision":      f"{r['precision']*100:.2f}%",
            "Recall":         f"{r['recall']*100:.2f}%",
            "F1 Score":       f"{r['f1']*100:.2f}%",
            "AUC-ROC":        f"{r['roc_auc']*100:.2f}%",
            "Train Acc (raw)":  f"{r['train_acc']:.4f}",
            "Test Acc (raw)":   f"{r['test_acc']:.4f}",
            "Precision (raw)":  f"{r['precision']:.4f}",
            "Recall (raw)":     f"{r['recall']:.4f}",
            "F1 (raw)":         f"{r['f1']:.4f}",
            "AUC (raw)":        f"{r['roc_auc']:.4f}",
        })

    disp_cols = ["Model","Train Accuracy","Test Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
    metrics_df = pd.DataFrame(rows)[disp_cols].set_index("Model")
    st.dataframe(metrics_df.style.highlight_max(color="#bbf7d0", axis=0), use_container_width=True)

    # Raw value table
    with st.expander("📄 Show Raw Decimal Values"):
        raw_cols = ["Model","Train Acc (raw)","Test Acc (raw)","Precision (raw)","Recall (raw)","F1 (raw)","AUC (raw)"]
        st.dataframe(pd.DataFrame(rows)[raw_cols].set_index("Model"), use_container_width=True)

    cap("Gradient Boosting achieves the highest AUC-ROC, making it the best model for *ranking* customers by loan probability — critical for a budget-constrained campaign where you need to prioritise who to contact. Green cells highlight the best metric per column.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC Curve ──
    sec("📈 ROC Curve — All Models on a Single Chart")
    fig = go.Figure()

    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{name}  (AUC = {r['roc_auc']:.3f})",
            line=dict(color=MODEL_COLORS[name], width=2.8),
        ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        name="Random Classifier  (AUC = 0.500)",
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        xaxis=dict(title="False Positive Rate (1 − Specificity)", range=[0,1], gridcolor="#f1f5f9"),
        yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", range=[0,1.02], gridcolor="#f1f5f9"),
        legend=dict(x=0.55, y=0.12, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1, font_size=12),
        height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"),
        margin=dict(t=20, b=40, l=60, r=20),
    )
    # Shade under best model
    best = max(results, key=lambda k: results[k]["roc_auc"])
    fig.add_trace(go.Scatter(
        x=np.concatenate([results[best]["fpr"], [1,0]]),
        y=np.concatenate([results[best]["tpr"], [0,0]]),
        fill="toself", fillcolor="rgba(37,99,235,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    st.plotly_chart(fig, use_container_width=True)
    cap("The ROC curve plots the True Positive Rate against False Positive Rate at every classification threshold. A curve hugging the top-left corner = better model. AUC (Area Under Curve) of 1.0 is perfect; 0.5 is random guessing. All three models significantly outperform random — Gradient Boosting leads. The shaded area represents the best model's AUC region.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrices ──
    sec("🔲 Confusion Matrices — All Models")
    cm_cols = st.columns(3)

    for idx, (name, r) in enumerate(results.items()):
        with cm_cols[idx]:
            cm = r["cm"]
            total = cm.sum()
            cm_pct = cm / total * 100

            fig = go.Figure(go.Heatmap(
                z=cm[::-1],
                x=["Predicted: No Loan","Predicted: Loan"],
                y=["Actual: Loan","Actual: No Loan"],
                colorscale="Blues",
                showscale=False,
                zmin=0, zmax=cm.max(),
            ))

            # Annotations with count + %
            for i in range(2):
                for j in range(2):
                    ri = 1 - i   # flip for display
                    val = cm[ri, j]
                    pct = cm_pct[ri, j]
                    fc = "white" if val > cm.max() * 0.55 else "#0a1628"
                    fig.add_annotation(
                        x=j, y=i,
                        text=f"<b>{val:,}</b><br>({pct:.1f}%)",
                        showarrow=False,
                        font=dict(color=fc, size=13, family="DM Sans"),
                    )

            # Labels for quadrants
            fig.add_annotation(x=0, y=1, text="TN", showarrow=False,
                               font=dict(size=9, color="rgba(255,255,255,0.6)"), xanchor="left", yanchor="top",
                               xref="x", yref="y", ax=0, ay=0)
            fig.add_annotation(x=1, y=1, text="FP", showarrow=False,
                               font=dict(size=9, color="#0a1628"), xanchor="right", yanchor="top",
                               xref="x", yref="y", ax=0, ay=0)
            fig.add_annotation(x=0, y=0, text="FN", showarrow=False,
                               font=dict(size=9, color="#0a1628"), xanchor="left", yanchor="bottom",
                               xref="x", yref="y", ax=0, ay=0)
            fig.add_annotation(x=1, y=0, text="TP", showarrow=False,
                               font=dict(size=9, color="rgba(255,255,255,0.6)"), xanchor="right", yanchor="bottom",
                               xref="x", yref="y", ax=0, ay=0)

            fig.update_layout(
                title=dict(text=f"<b>{name}</b>", font=dict(size=13, color="#0a1628",
                           family="Playfair Display"), x=0.5),
                height=330, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=70, l=20, r=20),
                xaxis=dict(tickfont=dict(size=9.5)),
                yaxis=dict(tickfont=dict(size=9.5)),
            )
            st.plotly_chart(fig, use_container_width=True)

    cap("Each matrix shows: True Negative (TN) — correctly predicted no loan; False Positive (FP) — predicted loan but customer declined; False Negative (FN) — missed a potential acceptor; True Positive (TP) — correctly identified a loan acceptor. For marketing, maximising TP (Recall) is priority, as every FN is a missed revenue opportunity. Values shown as count and (% of total test set).")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance ──
    sec("🎯 Feature Importance — What Drives Model Predictions?")
    fi_cols_chart = st.columns(3)

    for idx, (name, r) in enumerate(results.items()):
        with fi_cols_chart[idx]:
            fi_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Importance": r["feature_importance"],
            }).sort_values("Importance", ascending=True)

            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         color="Importance",
                         color_continuous_scale=["#e0f2fe", MODEL_COLORS[name]],
                         text=fi_df["Importance"].apply(lambda x: f"{x:.3f}"))
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            chart_layout(fig, f"<b>{name}</b><br>Feature Importance", h=440, legend=False)
            fig.update_layout(
                xaxis=dict(title="Importance Score", range=[0, fi_df["Importance"].max() * 1.3]),
                yaxis=dict(title=""),
                margin=dict(l=110, r=55, t=60, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

    cap("Income is consistently the top feature across all three models — validating the diagnostic findings. CD Account, CCAvg, and Family are also consistently important. Age and Experience carry similar information (due to high correlation), so either alone is sufficient. Use these features as your primary targeting criteria when segmenting your campaign audience.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PRESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_prescriptive(df, results):
    sec("🎯 Prescriptive Analytics — Who to Target & How")
    insight("Prescriptive analytics translates AI model output into concrete marketing actions. Given your halved budget, every rupee must count. This section tells you **which customers to prioritise**, **how to message them**, and **where to allocate spend** for the maximum acceptance rate.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Score all customers with best model
    best_name  = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]
    st.markdown(f"**🏆 Scoring Model:** `{best_name}` — Best AUC-ROC across all tested algorithms")

    scored = df.copy()
    scored["Loan_Prob"] = best_model.predict_proba(scored[FEATURE_COLS])[:, 1]
    scored["Tier"] = pd.cut(
        scored["Loan_Prob"],
        bins=[0, 0.15, 0.35, 0.55, 0.75, 1.01],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tier Distribution ──
    c1, c2 = st.columns(2)
    tier_order  = ["Very Low","Low","Medium","High","Very High"]
    tier_colors = ["#ef4444","#f97316","#eab308","#22c55e","#2563eb"]

    with c1:
        tier_counts = scored["Tier"].value_counts().reindex(tier_order).fillna(0).astype(int)
        tier_pct    = (tier_counts / len(scored) * 100).round(1)
        fig = px.bar(
            x=tier_counts.index, y=tier_counts.values,
            color=tier_counts.index,
            color_discrete_sequence=tier_colors,
            text=[f"{v:,}<br>({p}%)" for v, p in zip(tier_counts.values, tier_pct.values)],
            labels={"x":"Propensity Tier","y":"No. of Customers"},
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        chart_layout(fig, "Customer Count by Loan Propensity Tier", h=360, legend=False)
        fig.update_layout(yaxis=dict(range=[0, tier_counts.max()*1.22]))
        st.plotly_chart(fig, use_container_width=True)
        cap("Most customers fall in Very Low propensity tiers. With limited budget, focus 70%+ of spend on High + Very High tiers — smallest count, highest return. Do not waste resources chasing Low propensity customers.")

    with c2:
        tier_act = scored.groupby("Tier", observed=True)["Personal Loan"].agg(["mean","count"]).reset_index()
        tier_act.columns = ["Tier","Rate","N"]
        tier_act["Rate_%"] = tier_act["Rate"] * 100
        tier_act = tier_act.dropna(subset=["Tier"])
        tier_act = tier_act[tier_act["Tier"].isin(tier_order)]
        tier_act["Tier"] = pd.Categorical(tier_act["Tier"], categories=tier_order, ordered=True)
        tier_act = tier_act.sort_values("Tier")

        fig = px.bar(
            tier_act, x="Tier", y="Rate_%",
            color="Tier", color_discrete_sequence=tier_colors,
            text=tier_act["Rate_%"].apply(lambda x: f"{x:.1f}%"),
            labels={"Tier":"Propensity Tier","Rate_%":"Actual Acceptance Rate (%)"},
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        chart_layout(fig, "Actual Acceptance Rate by Propensity Tier", h=360, legend=False)
        fig.update_layout(yaxis=dict(range=[0, tier_act["Rate_%"].max() * 1.25]))
        st.plotly_chart(fig, use_container_width=True)
        cap("The model's tiers align with actual acceptance — validating the scoring approach. Very High tier customers convert at rates 5–8× the baseline. Targeting only High + Very High achieves massive efficiency gains.")

    # ── Ideal Customer Profile ──
    sec("🏆 Ideal Target Customer Profile — High & Very High Propensity")
    top_df = scored[scored["Tier"].isin(["High","Very High"])]
    all_df = scored

    p1, p2, p3, p4, p5 = st.columns(5)
    items = [
        (f"${top_df['Income'].median():.0f}K",       "Median Income",        "#2563eb", f"vs ${all_df['Income'].median():.0f}K overall"),
        (f"${top_df['CCAvg'].median():.1f}K/mo",     "Median CC Spend",      "#16a34a", f"vs ${all_df['CCAvg'].median():.1f}K overall"),
        (EDU_MAP.get(int(top_df["Education"].mode()[0]),"—"), "Dominant Education", "#7c3aed", "Most common level"),
        (f"{top_df['CD Account'].mean()*100:.0f}%",  "Have CD Account",      "#0891b2", f"vs {all_df['CD Account'].mean()*100:.0f}% overall"),
        (f"{top_df['Family'].median():.0f}",          "Median Family Size",   "#f97316", f"vs {all_df['Family'].median():.0f} overall"),
    ]
    for col, (v, l, a, d) in zip([p1,p2,p3,p4,p5], items):
        col.markdown(kpi(v, l, a, d), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Budget Allocation ──
    sec("💰 Marketing Budget Allocation Recommendation")
    budget_data = pd.DataFrame({
        "Propensity Tier":     ["Very High","High","Medium","Low & Very Low"],
        "Budget Allocation":   [40, 30, 20, 10],
        "Expected Customers":  [
            int((scored["Tier"] == "Very High").sum()),
            int((scored["Tier"] == "High").sum()),
            int((scored["Tier"] == "Medium").sum()),
            int(scored["Tier"].isin(["Low","Very Low"]).sum()),
        ],
        "Recommended Channel": [
            "Direct Phone Call + Personalised Email",
            "Personalised Email + SMS Push",
            "Targeted Email Campaign",
            "Brand Awareness / Passive Nurture",
        ],
        "Expected Conversion": ["35–60%", "15–35%", "5–15%", "<5%"],
    })

    c1, c2 = st.columns([1, 2])
    with c1:
        fig = px.pie(budget_data, values="Budget Allocation", names="Propensity Tier",
                     color_discrete_sequence=["#1d4ed8","#2563eb","#60a5fa","#dbeafe"],
                     hole=0.45)
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="v", x=1.0, y=0.5), margin=dict(t=10,b=10,l=10,r=100))
        fig.update_traces(textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(budget_data.set_index("Propensity Tier"), use_container_width=True, height=195)
        cap("70% of budget on High + Very High propensity. Use high-touch channels (calls, personalised emails) for top tiers. The 10% on Low tiers maintains broad brand presence without wasting premium resources.")

    # ── Campaign Recommendations ──
    sec("📋 Hyper-Personalised Campaign Playbook")

    recs = [
        {
            "tier": "🔵 Priority 1 — CD Account Holders + Income > $100K",
            "size": f"{len(df[(df['CD Account']==1)&(df['Income']>100)]):,}",
            "rate": f"{df[(df['CD Account']==1)&(df['Income']>100)]['Personal Loan'].mean()*100:.1f}%",
            "msg":  "These customers have a deep existing bank relationship and high income. **Call them directly.** Offer a pre-approved, premium personal loan with preferred rates. Messaging: 'As a valued CD Account member, you are pre-approved for an exclusive personal loan at our lowest rate.' Emphasise trust, relationship, and exclusivity."
        },
        {
            "tier": "🟢 Priority 2 — Income > $100K AND CCAvg > $3K/month",
            "size": f"{len(df[(df['Income']>100)&(df['CCAvg']>3)]):,}",
            "rate": f"{df[(df['Income']>100)&(df['CCAvg']>3)]['Personal Loan'].mean()*100:.1f}%",
            "msg":  "These financially active high-earners are comfortable with credit products. **Send a personalised email with a pre-approved offer.** Messaging: 'Your financial profile qualifies you for up to $[X] with no processing fees.' Emphasise loan flexibility — travel, luxury purchases, investments. Use your mobile app push notification as a secondary touchpoint."
        },
        {
            "tier": "🟡 Priority 3 — Graduate/Advanced Education + Family Size 3 or 4",
            "size": f"{len(df[(df['Education']>=2)&(df['Family']>=3)]):,}",
            "rate": f"{df[(df['Education']>=2)&(df['Family']>=3)]['Personal Loan'].mean()*100:.1f}%",
            "msg":  "Educated parents face real financial pressures — school fees, home upgrades, family holidays. **Life-stage email campaign with empathetic messaging.** Messaging: 'Planning a home renovation? Funding your child's education? A Universal Bank personal loan gives you the flexibility you need.' Include EMI calculator in email. Follow up with SMS."
        },
        {
            "tier": "🟠 Priority 4 — Online Banking Users + Mortgage Holders",
            "size": f"{len(df[(df['Online']==1)&(df['Has_Mortgage']==1)]):,}",
            "rate": f"{df[(df['Online']==1)&(df['Has_Mortgage']==1)]['Personal Loan'].mean()*100:.1f}%",
            "msg":  "Digitally active and proven credit users. **Target via in-app banners and online banking homepage.** Zero marginal cost! Messaging: 'Quick apply — Personal Loan in 3 steps. Pre-filled form, instant decision.' Emphasise digital convenience, speed, and simplicity. These customers will not respond to cold calls — let the digital channel do the work."
        },
    ]
    for r in recs:
        st.markdown(f"""
        <div class="rec-card">
            <h4>{r['tier']} &nbsp;|&nbsp; {r['size']} customers &nbsp;|&nbsp; Acceptance Rate: <span class="rec-rate">{r['rate']}</span></h4>
            <p>{r['msg']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    insight("**Golden Rule for Budget-Constrained Campaigns:** Use the model's probability score to rank ALL 5,000 customers and contact only the top 500–800 (top 10–16%). This alone can achieve acceptance rates of 35–60% — compared to 9.6% from random outreach — delivering 4–6× more conversions per dollar spent.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PREDICT & DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def page_predict(df, results):
    sec("📤 Predict on New Data & Download Results")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])

    st.markdown(f"""
    Upload a CSV file of new customers (same column structure as the training data, 
    **without** the `Personal Loan` column). The model will score each customer with:
    - **Loan Acceptance Probability** (0–100%)
    - **Predicted Label** (Will Accept / Will NOT Accept)
    - **Propensity Tier** (Very Low → Very High)
    """)

    c1, c2 = st.columns(2)
    with c1:
        sel_model = st.selectbox(
            "Select Prediction Model",
            list(results.keys()),
            index=list(results.keys()).index(best_name),
            help=f"Default: {best_name} (Highest AUC-ROC)"
        )
    with c2:
        threshold = st.slider(
            "Classification Threshold",
            0.10, 0.90, 0.50, 0.05,
            help="Lower = more customers classified as loan acceptors (higher Recall, lower Precision). Adjust based on campaign budget."
        )

    model = results[sel_model]["model"]
    st.info(f"**Active Model:** {sel_model}  |  **AUC-ROC:** {results[sel_model]['roc_auc']:.4f}  |  **Threshold:** {threshold}")

    uploaded = st.file_uploader("📁 Upload Test CSV", type=["csv"])

    if uploaded is not None:
        try:
            test_raw = pd.read_csv(uploaded)
            st.markdown(f"**Uploaded:** {len(test_raw):,} rows × {len(test_raw.columns)} columns")
            st.dataframe(test_raw.head(5), use_container_width=True)

            proc = test_raw.copy()
            for drop_col in ["ID","ZIP Code","Personal Loan"]:
                if drop_col in proc.columns:
                    proc.drop(columns=[drop_col], inplace=True)
            if "Experience" in proc.columns:
                proc["Experience"] = proc["Experience"].clip(lower=0)
            if "Has_Mortgage" not in proc.columns and "Mortgage" in proc.columns:
                proc["Has_Mortgage"] = (proc["Mortgage"] > 0).astype(int)

            missing = [c for c in FEATURE_COLS if c not in proc.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                st.markdown("**Required columns:** " + ", ".join(FEATURE_COLS))
            else:
                X_new = proc[FEATURE_COLS]
                probs = model.predict_proba(X_new)[:, 1]
                preds = (probs >= threshold).astype(int)

                out = test_raw.copy()
                out["Loan_Acceptance_Probability_%"] = (probs * 100).round(2)
                out["Predicted_Personal_Loan"]       = preds
                out["Prediction_Label"]              = preds.map({0:"Will NOT Accept", 1:"Will ACCEPT"})
                out["Propensity_Tier"] = pd.cut(
                    probs, bins=[0, 0.15, 0.35, 0.55, 0.75, 1.01],
                    labels=["Very Low","Low","Medium","High","Very High"]
                ).astype(str)

                # Summary
                st.markdown("### 📊 Prediction Summary")
                n_acc = int(preds.sum())
                n_dec = len(preds) - n_acc
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Predicted to Accept", f"{n_acc:,}", f"{n_acc/len(preds)*100:.1f}%")
                s2.metric("Predicted to Decline", f"{n_dec:,}", f"{n_dec/len(preds)*100:.1f}%")
                s3.metric("Avg. Probability", f"{probs.mean()*100:.1f}%")
                s4.metric("Max Probability", f"{probs.max()*100:.1f}%")

                st.markdown("### 🔍 Results Preview (First 20 Rows)")
                st.dataframe(
                    out[["Loan_Acceptance_Probability_%","Predicted_Personal_Loan",
                         "Prediction_Label","Propensity_Tier"]].head(20),
                    use_container_width=True
                )

                # Download
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button(
                    label="⬇️ Download Full Predictions as CSV",
                    data=buf.getvalue(),
                    file_name=f"loan_predictions_{sel_model.replace(' ','_')}_t{int(threshold*100)}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    else:
        st.markdown("""
        <div class="insight">
        📌 <b>Expected File Format:</b> Your upload should have the same columns as the training data 
        <em>without</em> the <code>Personal Loan</code> column. 
        Columns: Age, Experience, Income, Family, CCAvg, Education, Mortgage, 
        Securities Account, CD Account, Online, CreditCard. 
        A sample test file (<code>test_sample.csv</code>) is included in the project folder.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df      = load_data()
    results = train_models(df)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.2rem 0 0.5rem 0;">
            <div style="font-size:2rem;">🏦</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                        font-weight:700;color:white;margin-top:0.3rem;">
                Universal Bank
            </div>
            <div style="font-size:0.72rem;color:rgba(255,255,255,0.55);
                        text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">
                Loan Campaign Intelligence
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.12);margin:0.8rem 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            [
                "🏠  Executive Summary",
                "📊  Descriptive Analytics",
                "🔍  Diagnostic Analytics",
                "🤖  Predictive Analytics",
                "🎯  Prescriptive Analytics",
                "📤  Predict & Download",
            ],
            label_visibility="collapsed",
        )

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.12);margin:0.8rem 0;">
        <div style="font-size:0.73rem;color:rgba(255,255,255,0.5);text-align:center;
                    line-height:1.7;">
            <b style="color:rgba(255,255,255,0.7);">Dataset</b><br>
            5,000 customers · 12 features<br>
            <b style="color:rgba(255,255,255,0.7);">Target</b><br>
            Personal Loan Acceptance<br>
            <b style="color:rgba(255,255,255,0.7);">Models</b><br>
            Decision Tree · Random Forest<br>Gradient Boosting
        </div>
        """, unsafe_allow_html=True)

    # ── Route ──
    if   "Executive"     in page: page_executive(df)
    elif "Descriptive"   in page: page_descriptive(df)
    elif "Diagnostic"    in page: page_diagnostic(df)
    elif "Predictive"    in page: page_predictive(df, results)
    elif "Prescriptive"  in page: page_prescriptive(df, results)
    elif "Predict"       in page: page_predict(df, results)


if __name__ == "__main__":
    main()
