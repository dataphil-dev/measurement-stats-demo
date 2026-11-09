"""Measurement comparison demo: did the change make a real difference?

DataPhil demo - the greenhouse trial data is made up. Upload your own CSV to try it; it is
processed in memory only and never stored.
Run locally:  pip install -r requirements.txt && streamlit run app.py
"""
import html
import io
import re
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


# ---------------------------------- look & feel (same palette as dataphil.dev) ----------------------------------
NAVY, INK, BODY, MUTED = "#1E3A6E", "#0E1A2F", "#3D4047", "#5A5E66"
GREY, LIGHT, PAPER, WARN = "#9AA3B2", "#D3D9E3", "#F4F5F8", "#B4531A"

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Instrument+Serif&display=swap');
html, body, .stApp, .stMarkdown, p, label, button, input, textarea, select, li {font-family:'IBM Plex Sans',system-ui,sans-serif;}
.block-container {padding-top:2rem; max-width:1200px;}
h1 {font-family:'Instrument Serif',Georgia,serif !important; font-weight:400 !important; font-size:2.9rem !important;
    color:#0E1A2F !important; letter-spacing:-.01em; line-height:1.05 !important; padding:0 !important;}
.dp-bar {font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:#C3CCDB; background:#0E1A2F;
    padding:.55rem 1rem; border-radius:10px; display:flex; justify-content:space-between; margin-bottom:1.1rem;}
.dp-bar b {font-weight:400; color:#fff; background:#17253F; border-radius:20px; padding:0 .6rem;}
.dp-lead {color:#5A5E66; font-size:1rem; margin:.2rem 0 1.2rem;}
.dp-kpi {background:#fff; border:1px solid #D3D9E3; border-radius:12px; padding:.85rem 1rem; height:100%;}
.dp-kpi .l {font-size:.82rem; color:#5A5E66;}
.dp-kpi .v {font-size:1.75rem; font-weight:500; color:#0E1A2F; line-height:1.25; margin:.15rem 0;}
.dp-kpi .c {font-size:.8rem; color:#5A5E66;} .dp-kpi .c.warn {color:#B4531A;}
.dp-msg {background:#0E1A2F; color:#fff; border-radius:12px; padding:1rem 1.25rem; margin:1rem 0 1.4rem; font-size:1.02rem; line-height:1.5;}
.dp-msg .k {display:block; font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.14em; color:#9DB6E3; margin-bottom:.2rem;}
.dp-h {font-weight:600; font-size:1.08rem; color:#0E1A2F; margin:.6rem 0 0;}
.dp-sub {font-size:.84rem; color:#5A5E66; margin:0 0 .4rem;}
.dp-steps {display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:1.4rem;}
.dp-steps div {border:1px solid #D3D9E3; border-radius:12px; padding:.9rem 1rem; font-size:.88rem; color:#3D4047; background:#fff;}
.dp-steps div.r {background:#1E3A6E; color:#fff; border-color:#1E3A6E;}
.dp-steps span {display:block; font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.14em; color:#5A5E66; margin-bottom:.3rem;}
.dp-steps div.r span {color:#9DB6E3;}
.dp-src {font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:#3D4047; line-height:1.9;}
.dp-src i {color:#2E7D4F; font-style:normal; margin-right:.4rem;}
.dp-note {background:#E7EBF2; border-radius:10px; padding:.7rem .9rem; font-size:.8rem; color:#3D4047; margin-top:1rem;}
@media (max-width:700px){ .dp-steps {grid-template-columns:1fr;} h1 {font-size:2.2rem !important;} }
</style>"""


def esc(s) -> str:
    return html.escape(str(s))


def bar(left: str, right: str = "Sample data"):
    st.markdown(f'<div class="dp-bar"><span>{esc(left)}</span><b>{esc(right)}</b></div>', unsafe_allow_html=True)


def kpi(col, label, value, context="", warn=False):
    col.markdown(f'<div class="dp-kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
                 f'<div class="c{" warn" if warn else ""}">{"⚠ " if warn else ""}{esc(context)}</div></div>',
                 unsafe_allow_html=True)


def message(text_html: str, label="MESSAGE FOR THE MEETING"):
    st.markdown(f'<div class="dp-msg"><span class="k">{label}</span>{text_html}</div>', unsafe_allow_html=True)


def heading(title: str, sub: str = ""):
    st.markdown(f'<p class="dp-h">{esc(title)}</p>' + (f'<p class="dp-sub">{esc(sub)}</p>' if sub else ""),
                unsafe_allow_html=True)


def sources(files):
    st.markdown("<p class='dp-sub' style='letter-spacing:.12em;font-family:IBM Plex Mono'>DATA SOURCES</p>"
                + "".join(f"<div class='dp-src'><i>✓</i>{esc(f)}</div>" for f in files), unsafe_allow_html=True)


def steps(before: str, tool: str, result: str):
    st.markdown(f'<div class="dp-steps"><div><span>1 · BEFORE</span>{before}</div><div><span>2 · THE TOOL</span>{tool}'
                f'</div><div class="r"><span>3 · RESULT</span>{result}</div></div>', unsafe_allow_html=True)


def chart_style(chart):
    return (chart.properties(padding={"left": 8, "right": 8, "top": 4, "bottom": 4}).configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, labelFont="IBM Plex Sans", titleFont="IBM Plex Sans",
                            labelFontSize=12, titleFontSize=12, gridColor="#EEF1F5", domainColor=LIGHT, tickColor=LIGHT)
            .configure_text(font="IBM Plex Sans"))


# ---------------------------------- UI ----------------------------------
def verdict(lo, hi, mrd, unit, other, base):
    """Turn the interval into a decision, compared with the smallest difference that matters."""
    m = f"{mrd:g} {unit}".strip()
    if lo > 0 and lo >= mrd:
        return "good", (f"<b>{esc(other)} is clearly better.</b> Even the low end of the range is above the "
                        f"{m} that matters.")
    if lo > 0 and hi >= mrd:
        return "good", (f"<b>{esc(other)} is better</b>, and probably by a relevant amount, but the low end of the "
                        f"range is below {m}. A few more measurements would settle it.")
    if lo > 0:
        return "small", (f"<b>The difference is real but small:</b> the whole range is below the {m} that matters.")
    if hi < 0:
        return "worse", f"<b>{esc(other)} is worse than {esc(base)}.</b>"
    return "unclear", (f"<b>Not proven.</b> The range includes zero, so the data cannot tell {esc(other)} and "
                       f"{esc(base)} apart yet.")


st.set_page_config(page_title="Measurement comparison (demo)", page_icon="📏", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    up = st.file_uploader("Your own CSV (optional): one group column, one number column", type=["csv"])
    sources([up.name if up else SAMPLE.name])
    df = pd.read_csv(up if up else SAMPLE, sep=None, engine="python")
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and 2 <= df[c].nunique() <= 20]
    if not num_cols or not cat_cols:
        st.error("Need at least one number column and one group column.")
        st.stop()
    st.markdown("<p class='dp-h' style='margin-top:1.2rem'>Question</p>", unsafe_allow_html=True)
    value = st.selectbox("Measured value", num_cols, index=len(num_cols) - 1)
    group = st.selectbox("Group column", cat_cols)
    levels = df[group].dropna().unique().tolist()
    base = st.selectbox("Baseline", levels, index=0)
    other = st.selectbox("Compared with", [g for g in levels if g != base])
    unit = st.text_input("Unit", "cm" if not up else "")
    mrd = st.number_input("Smallest difference that matters", 0.0, value=2.0 if not up else 0.0, step=0.5,
                          help="Agree on this before looking at the result. It turns 'significant' into a decision.")
    with st.expander("Settings"):
        level = st.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95,
                                 format_func=lambda x: f"{x:.0%}")
        drop_out = st.checkbox("Exclude outliers (1.5 × IQR rule)", False)
    st.markdown("<div class='dp-note'>Runs in your browser. Your file is not uploaded anywhere and not stored."
                "</div>", unsafe_allow_html=True)

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
lo, hi = r["ci"]
kind, text = verdict(lo, hi, mrd, unit, other, base)
u = f" {unit}" if unit else ""

bar("lab-report.local · trial comparison")
st.title("Did the change make a real difference?")
st.markdown(f"<p class='dp-lead'>{'Greenhouse trial (fictional): plant height with a new vs. the standard fertiliser, ' if not up else ''}"
            f"{len(a)} vs. {len(b)} measurements. The answer comes as a range, not a single number.</p>",
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, f"{base} (baseline)", f"{a.mean():.1f}{u}", f"mean of {len(a)} measurements")
kpi(k2, f"{other}", f"{b.mean():.1f}{u}", f"{r['diff']:+.1f}{u} ({r['rel']:+.0%}) vs. baseline")
kpi(k3, f"Likely difference ({level:.0%})", f"{lo:+.1f} to {hi:+.1f}{u}",
    "range includes zero" if lo <= 0 <= hi else f"effect size {r['d']:.2f} ({size_word(r['d'])})",
    warn=lo <= 0 <= hi)
kpi(k4, "Outliers", str(n_out), ("excluded from the result" if drop_out else "included; see the chart")
    if n_out else "none found", warn=n_out > 0 and not drop_out)

extra = ""
if n_out and not drop_out:
    a2 = data.loc[(data[group] == base) & ~data["outlier"], value].to_numpy()
    b2 = data.loc[(data[group] == other) & ~data["outlier"], value].to_numpy()
    if len(a2) >= 3 and len(b2) >= 3:
        r2 = analyse(a2, b2, level)
        extra = (f" Without the {n_out} outlier{'s' if n_out > 1 else ''} the difference would be {r2['diff']:+.1f}{u} "
                 f"({r2['ci'][0]:+.1f} to {r2['ci'][1]:+.1f}), so the conclusion "
                 f"{'does not change' if verdict(*r2['ci'], mrd, unit, other, base)[0] == kind else 'changes'}.")
message(f"{text} Measured difference: <b>{r['diff']:+.1f}{u}</b>, likely between {lo:+.1f} and {hi:+.1f}{u}."
        + esc(extra), "ANSWER FOR THE DECISION")

left, right = st.columns([1, 1], gap="large")
with left:
    heading("The likely difference vs. what matters",
            f"{level:.0%} range for {other} minus {base}; dashed line = smallest difference that matters")
    span = max(abs(lo), abs(hi), mrd, 0.1) * 1.35
    iv = pd.DataFrame({"lo": [lo], "hi": [hi], "d": [r["diff"]], "y": ["difference"]})
    col = NAVY if kind == "good" else (WARN if kind in ("worse", "unclear") else GREY)
    xs = alt.Scale(domain=[min(-span * 0.4, lo * 1.2), span])
    rng = alt.Chart(iv).mark_rule(strokeWidth=10, color=col, opacity=0.35, strokeCap="round").encode(
        x=alt.X("lo:Q", scale=xs, title=f"difference{u}", axis=alt.Axis(grid=False)), x2="hi:Q", y=alt.Y("y:N", axis=None))
    pt = alt.Chart(iv).mark_point(filled=True, size=160, color=col, opacity=1).encode(x="d:Q", y="y:N")
    lab = alt.Chart(iv).mark_text(dy=24, fontSize=13, color=INK, fontWeight=500).encode(
        x="d:Q", y="y:N", text=alt.Text("lab:N")).transform_calculate(
        lab=f"format(datum.d, '+.1f') + '{u}  (' + format(datum.lo, '+.1f') + ' to ' + format(datum.hi, '+.1f') + ')'")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GREY).encode(x="x:Q")
    layers = [zero, rng, pt, lab]
    if mrd > 0:
        mdf = pd.DataFrame({"x": [mrd], "t": [f"matters from {mrd:+g}{u}"]})
        layers += [alt.Chart(mdf).mark_rule(color=WARN, strokeDash=[5, 4]).encode(x="x:Q"),
                   alt.Chart(mdf).mark_text(align="left", dx=5, dy=-48, color=WARN, fontSize=12).encode(
                       x="x:Q", text="t:N")]
    zl = alt.Chart(pd.DataFrame({"x": [0], "t": ["no difference"]})).mark_text(
        align="right", dx=-5, dy=-48, color=MUTED, fontSize=12).encode(x="x:Q", text="t:N")
    st.altair_chart(chart_style(alt.layer(*layers, zl).properties(height=150)), use_container_width=True)
    st.caption("If the whole bar lies right of the dashed line, the change is worth it. "
               "If it crosses zero, the data cannot tell the groups apart.")
with right:
    heading(f"All measurements: {other} vs. {base}", "Each dot is one measurement, the bar is the group mean"
            + ("; orange = outlier" if n_out else ""))
    data["kind"] = np.where(data["outlier"], "outlier", np.where(data[group] == other, "compared", "baseline"))
    pts = alt.Chart(data).mark_circle(size=60, opacity=0.8).encode(
        y=alt.Y(f"{group}:N", sort=[other, base], title=None, axis=alt.Axis(ticks=False, domain=False,
                                                                           labelPadding=8)),
        yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-1.5, 2.5])),
        x=alt.X(f"{value}:Q", scale=alt.Scale(zero=False), title=f"{value}{' (' + unit + ')' if unit else ''}"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["baseline", "compared", "outlier"],
                                                  range=[GREY, NAVY, WARN]), legend=None),
        tooltip=list(data.columns.drop(["outlier", "kind"]))).transform_calculate(jitter="random()")
    mdf = used.groupby(group)[value].mean().reset_index()
    means = alt.Chart(mdf).mark_tick(color=INK, thickness=3, size=34).encode(
        y=alt.Y(f"{group}:N", sort=[other, base]), x=f"{value}:Q")
    mlab = alt.Chart(mdf).mark_text(dy=-24, color=INK, fontSize=12).encode(
        y=alt.Y(f"{group}:N", sort=[other, base]), x=f"{value}:Q",
        text=alt.Text(f"{value}:Q", format=".1f"))
    st.altair_chart(chart_style((pts + means + mlab).properties(height=230)), use_container_width=True)

with st.expander("For the statistician: methods and checks"):
    summary = used.groupby(group)[value].agg(["count", "mean", "std", "min", "median", "max"]).round(2)
    st.dataframe(summary, use_container_width=True)
    rows = [
        ("Test", "Welch t-test"),
        ("p-value", f"{r['p']:.4f}"),
        (f"{level:.0%} CI of difference (t)", f"{lo:.2f} to {hi:.2f}"),
        (f"{level:.0%} CI of difference (bootstrap)", f"{r['boot'][0]:.2f} to {r['boot'][1]:.2f}"),
        ("Effect size (Cohen's d)", f"{r['d']:.2f} ({size_word(r['d'])})"),
        ("Mann-Whitney U p-value", f"{r['p_mw']:.4f}"),
        ("Both groups roughly normal (Shapiro-Wilk)", "yes" if r["normal"] else "no, rely more on bootstrap"),
        ("Outliers excluded", str(n_out) if drop_out else "0"),
        ("Smallest difference that matters", f"{mrd:g}{u}"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Check", "Value"]), hide_index=True, use_container_width=True)
    st.caption("This demo assumes independent measurements in two groups. In a real project the first step is to "
               "check how the data were collected (randomisation, repeated measurements, batches).")

st.download_button("Download Excel report (answer, checks, data)",
                   excel_report(used.drop(columns="outlier"), summary,
                                rows + [("Answer", re.sub("<[^>]+>", "", text))]),
                   file_name="comparison_report.xlsx", type="primary")

steps("A trial ends with two columns of numbers and a p-value nobody outside the lab can act on.",
      "Compares the groups, checks the assumptions and states the result as a range next to what matters.",
      "A yes/no/unclear answer you can defend. Try it: set the <b>smallest difference</b> to 4 cm.")
