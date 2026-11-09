"""Measurement comparison demo: did the change make a real difference?

DataPhil demo - the greenhouse trial data is made up. Upload your own CSV to try it; it is
processed in memory only and never stored.
Run locally:  pip install -r requirements.txt && streamlit run app.py
"""
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

SAMPLE = Path(__file__).parent / "data" / "greenhouse_trial.csv"
RNG = np.random.default_rng(42)


def bootstrap_ci(a, b, n=5000, level=0.95):
    """Confidence interval for mean(b) - mean(a) by resampling both groups."""
    diffs = np.array([RNG.choice(b, len(b)).mean() - RNG.choice(a, len(a)).mean() for _ in range(n)])
    lo, hi = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return lo, hi


def outliers(x: pd.Series) -> pd.Series:
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)


def analyse(a: np.ndarray, b: np.ndarray, level: float):
    diff = b.mean() - a.mean()
    t = stats.ttest_ind(b, a, equal_var=False)
    ci = t.confidence_interval(level)
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = diff / pooled_sd
    mw = stats.mannwhitneyu(b, a, alternative="two-sided")
    sw_a, sw_b = stats.shapiro(a).pvalue, stats.shapiro(b).pvalue
    boot = bootstrap_ci(a, b, level=level)
    return {"diff": diff, "rel": diff / a.mean(), "ci": (ci.low, ci.high), "p": t.pvalue, "d": d,
            "p_mw": mw.pvalue, "normal": min(sw_a, sw_b) > 0.05, "boot": boot}


def size_word(d):
    d = abs(d)
    return "very small" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def plain_language(r, base, other, unit, level):
    lo, hi = r["ci"]
    pct = f"{level:.0%}"
    if lo > 0 or hi < 0:
        direction = "higher" if r["diff"] > 0 else "lower"
        verdict = (f"**{other}** is on average **{abs(r['diff']):.1f} {unit} {direction}** than **{base}** "
                   f"({r['rel']:+.0%}). With {pct} confidence the true difference lies between "
                   f"{lo:.1f} and {hi:.1f} {unit}, so it is very unlikely to be chance.")
    else:
        verdict = (f"The data **do not show a clear difference** between **{other}** and **{base}**. "
                   f"The true difference could be anywhere between {lo:.1f} and {hi:.1f} {unit}. "
                   "More measurements would narrow this down.")
    extra = f" The effect is {size_word(r['d'])} (Cohen's d = {r['d']:.2f})."
    check = (" A rank-based test that does not assume normal data agrees."
             if (r["p_mw"] < 0.05) == (r["p"] < 0.05) else
             " Note: a rank-based test disagrees, so treat this result with care.")
    return verdict + extra + check


def excel_report(df, summary, result_rows) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(result_rows, columns=["Result", "Value"]).to_excel(xw, sheet_name="Result", index=False)
        summary.to_excel(xw, sheet_name="Summary by group")
        df.to_excel(xw, sheet_name="Data", index=False)
    return buf.getvalue()


# ---------------------------------- UI ----------------------------------
st.set_page_config(page_title="Measurement comparison (demo)", page_icon="📏", layout="wide")
st.title("Did the change make a real difference?")
st.caption("DataPhil demo · compares two groups of measurements and explains the result in plain language · "
           "sample data is made up")

with st.sidebar:
    st.header("1  Data")
    up = st.file_uploader("CSV with one group column and one value column", type=["csv"])
    if not up:
        st.write("No file? A made-up greenhouse trial is used: plant height with a new vs a standard fertiliser.")
    df = pd.read_csv(up if up else SAMPLE, sep=None, engine="python")
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and 2 <= df[c].nunique() <= 20]
    if not num_cols or not cat_cols:
        st.error("Need at least one number column and one group column.")
        st.stop()
    st.header("2  Settings")
    value = st.selectbox("Measured value", num_cols, index=len(num_cols) - 1)
    group = st.selectbox("Group column", cat_cols)
    levels = df[group].dropna().unique().tolist()
    base = st.selectbox("Baseline group", levels, index=0)
    other = st.selectbox("Compare with", [g for g in levels if g != base])
    unit = st.text_input("Unit", "cm" if not up else "")
    level = st.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95, format_func=lambda x: f"{x:.0%}")
    drop_out = st.checkbox("Exclude outliers (1.5 × IQR rule)", False)

data = df[df[group].isin([base, other])].dropna(subset=[value]).copy()
data["outlier"] = data.groupby(group)[value].transform(outliers)
n_out = int(data["outlier"].sum())
used = data[~data["outlier"]] if drop_out else data
a = used.loc[used[group] == base, value].to_numpy()
b = used.loc[used[group] == other, value].to_numpy()
if len(a) < 3 or len(b) < 3:
    st.error("Each group needs at least 3 measurements.")
    st.stop()
r = analyse(a, b, level)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Mean {base} (n = {len(a)})", f"{a.mean():.1f} {unit}")
c2.metric(f"Mean {other} (n = {len(b)})", f"{b.mean():.1f} {unit}")
c3.metric("Difference", f"{r['diff']:+.1f} {unit}", f"{r['rel']:+.0%}")
c4.metric(f"{level:.0%} interval", f"{r['ci'][0]:+.1f} to {r['ci'][1]:+.1f}")

st.subheader("Result in plain language")
box = st.success if r["p"] < 0.05 else st.info
box(plain_language(r, base, other, unit, level))
if n_out:
    st.warning(f"{n_out} measurement(s) look like outliers (marked red in the chart). "
               + ("They are excluded." if drop_out else "They are included; tick the box in the sidebar to "
                                                        "see how much they matter."))

left, right = st.columns([3, 2])
with left:
    st.subheader("All measurements")
    pts = alt.Chart(data).mark_circle(size=70, opacity=0.75).encode(
        x=alt.X(f"{group}:N", sort=[base, other], title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("jitter:Q", scale=alt.Scale(domain=[-1.2, 2.2])),
        y=alt.Y(f"{value}:Q", scale=alt.Scale(zero=False), title=f"{value} ({unit})" if unit else value),
        color=alt.condition("datum.outlier", alt.value("#d1342f"), alt.value("#14a800")),
        tooltip=list(data.columns.drop("outlier")),
    ).transform_calculate(jitter="random()")
    means = alt.Chart(used).mark_tick(color="#262730", thickness=3, size=60).encode(
        x=alt.X(f"{group}:N", sort=[base, other]), y=f"mean({value}):Q")
    st.altair_chart((pts + means).properties(height=380), width="stretch")
    st.caption("Each dot is one measurement, the black bar is the group mean.")
with right:
    st.subheader("Details")
    summary = used.groupby(group)[value].agg(["count", "mean", "std", "min", "median", "max"]).round(2)
    st.dataframe(summary, width="stretch")
    rows = [
        ("Test", "Welch t-test"),
        ("p-value", f"{r['p']:.4f}"),
        (f"{level:.0%} CI of difference (t)", f"{r['ci'][0]:.2f} to {r['ci'][1]:.2f}"),
        (f"{level:.0%} CI of difference (bootstrap)", f"{r['boot'][0]:.2f} to {r['boot'][1]:.2f}"),
        ("Effect size (Cohen's d)", f"{r['d']:.2f} ({size_word(r['d'])})"),
        ("Mann-Whitney U p-value", f"{r['p_mw']:.4f}"),
        ("Both groups roughly normal (Shapiro-Wilk)", "yes" if r["normal"] else "no, rely more on bootstrap"),
        ("Outliers excluded", str(n_out) if drop_out else "0"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Check", "Value"]), hide_index=True, width="stretch")

st.subheader("3  Download")
st.download_button("Download Excel report (result, summary, data)",
                   excel_report(used.drop(columns="outlier"), summary, rows + [("Conclusion", plain_language(
                       r, base, other, unit, level).replace("**", ""))]),
                   file_name="comparison_report.xlsx", type="primary")
