import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="Unicorn Companies Explorer", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def load_data():
    df = pd.read_csv("data/Unicorn_Companies_cleaned.csv")
    df.loc[df["Total Raised (M)"] < 5, "Total Raised (M)"] = np.nan
    df["log_val"] = np.log(df["Valuation"])
    df["log_raised"] = np.log(df["Total Raised (M)"])
    df["Raised per Investor"] = df["Total Raised (M)"] / df["Investors Count"]
    df["Is US"] = (df["Country"] == "United States").astype(int)
    return df


df = load_data()
industries = sorted(df["Industry"].unique())
top_countries = list(df["Country"].value_counts().head(5).index)
NUM = ["log_raised", "Investors Count", "Deal Terms", "Founded Year",
       "Years to Unicorn", "Join Year", "Raised per Investor", "Is US"]


def build_X(d):
    X = d[NUM].copy()
    cg = d["Country"].where(d["Country"].isin(top_countries), "Other")
    for i in industries:
        X["ind_" + i] = (d["Industry"] == i).astype(float)
    for c in top_countries + ["Other"]:
        X["cty_" + c] = (cg == c).astype(float)
    return X.astype(float)


@st.cache_resource
def train_models():
    X = build_X(df)
    reg = make_pipeline(SimpleImputer(strategy="median"),
                        RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                              random_state=42, n_jobs=-1))
    reg.fit(X, df["log_val"])
    Xc = X.copy()
    Xc["log_val"] = df["log_val"]
    clf = make_pipeline(SimpleImputer(strategy="median"),
                        RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                               class_weight="balanced",
                                               random_state=42, n_jobs=-1))
    clf.fit(Xc, df["Exited"])
    return reg, clf, X.columns.tolist()


st.title("Unicorn Companies Explorer")
st.caption("1,037 private startups valued at $1B+  |  EDA, statistics and machine learning")

tab1, tab2, tab3 = st.tabs(["Overview", "Explore", "Predict"])

# ---------------- Overview ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{len(df):,}")
    c2.metric("Median valuation", f"${df['Valuation'].median():.2f}B")
    c3.metric("Top industry", df["Industry"].value_counts().idxmax())
    c4.metric("Exited", f"{int(df['Exited'].sum())} ({df['Exited'].mean()*100:.1f}%)")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        df["Industry"].value_counts().plot(kind="barh", ax=ax, color="#4C72B0")
        ax.invert_yaxis(); ax.set_ylabel(""); ax.set_title("Unicorns by industry")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["log_val"], bins=35, ax=ax, color="#55A868")
        ax.set_xlabel("log(Valuation in $B)"); ax.set_title("Valuation distribution (log scale)")
        st.pyplot(fig)

    st.subheader("Key findings")
    st.markdown("""
- **Funding drives valuation:** Total Raised is about 61% of the model's feature importance (Spearman 0.57).
- **Industry barely matters for valuation:** ANOVA p = 0.79, eta-squared = 0.009.
- **Exit rates differ by industry:** permutation test p = 0.023. Only 49 of 1,037 companies exited.
- **Models:** valuation CV R2 about 0.45; exit classifier CV AUC about 0.83.
""")

# ---------------- Explore ----------------
with tab2:
    f1, f2 = st.columns(2)
    sel_ind = f1.multiselect("Industry", industries)
    sel_cty = f2.multiselect("Country", sorted(df["Country"].unique()))
    sub = df.copy()
    if sel_ind:
        sub = sub[sub["Industry"].isin(sel_ind)]
    if sel_cty:
        sub = sub[sub["Country"].isin(sel_cty)]
    st.write(f"**{len(sub)}** companies  |  median valuation **${sub['Valuation'].median():.2f}B**"
             if len(sub) else "No companies match these filters.")
    if len(sub):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.scatterplot(data=sub, x="Total Raised (M)", y="Valuation", hue="Exited", alpha=0.7, ax=ax)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title("Total raised vs valuation (log-log)")
        st.pyplot(fig)
        st.dataframe(sub[["Company", "Valuation", "Country", "Industry", "Total Raised (M)",
                          "Investors Count"]].sort_values("Valuation", ascending=False),
                     use_container_width=True, hide_index=True)

# ---------------- Predict ----------------
with tab3:
    st.write("Enter a company profile to get an estimated valuation and exit likelihood.")
    a, b, c = st.columns(3)
    raised = a.number_input("Total raised ($M)", 5.0, 10000.0, 300.0, step=10.0)
    investors = b.number_input("Number of investors", 1, 100, 10)
    deals = c.number_input("Deal terms", 1, 20, 3)
    d1, d2, d3 = st.columns(3)
    founded = d1.number_input("Founded year", 1990, 2022, 2015)
    join_year = d2.number_input("Year became unicorn", 2008, 2022, 2021)
    country = d3.selectbox("Country", top_countries + ["Other"])
    industry = st.selectbox("Industry", industries)

    if st.button("Predict", type="primary"):
        reg, clf, cols = train_models()
        row = pd.DataFrame([{
            "log_raised": np.log(raised), "Investors Count": investors, "Deal Terms": deals,
            "Founded Year": founded, "Years to Unicorn": join_year - founded, "Join Year": join_year,
            "Raised per Investor": raised / investors, "Is US": int(country == "United States"),
            "Country": country, "Industry": industry}])
        X1 = build_X(row)[cols]
        logv = reg.predict(X1)[0]
        est = np.exp(logv)
        lo, hi = est / 1.75, est * 1.75
        X1c = X1.copy()
        X1c["log_val"] = logv
        p_exit = clf.predict_proba(X1c)[0, 1]

        m1, m2 = st.columns(2)
        m1.metric("Estimated valuation", f"${est:.2f}B")
        m1.caption(f"Typical range: \\${lo:.2f}B to \\${hi:.2f}B (model error is about 1.75x)")
        m2.metric("Exit likelihood score", f"{p_exit*100:.0f}%")
        m2.caption("Baseline exit rate is 4.7%. This score is not a calibrated probability, "
                   "because the model uses class weights.")
    st.info("Educational model. Valuation R2 is about 0.45, so treat results as rough estimates, "
            "not investment advice.")