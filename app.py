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
.block-container {padding-top:2.2rem; max-width:1180px;}
h1 {font-family:'Instrument Serif',Georgia,serif !important; font-weight:400 !important; font-size:2.8rem !important;
    color:#0E1A2F !important; letter-spacing:-.01em; line-height:1.05 !important; padding:0 !important;}
.dp-lead {color:#5A5E66; font-size:1.02rem; margin:.35rem 0 1.5rem; max-width:48rem;}
.dp-lead b {color:#0E1A2F; font-weight:500;}
.dp-chip {font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.08em; color:#1E3A6E;
    background:#E7EBF2; border-radius:20px; padding:.15rem .6rem; margin-left:.5rem; vertical-align:middle; white-space:nowrap;}
.dp-step {font-family:'IBM Plex Mono',monospace; font-size:.72rem !important; letter-spacing:.14em; color:#5A5E66; margin:0 0 .5rem;}
.dp-answer {background:#0E1A2F; color:#fff; border-radius:14px; padding:1.15rem 1.4rem 1.25rem; margin-bottom:.9rem;}
.dp-answer .k {font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.14em; color:#9DB6E3;}
.dp-answer .big {font-family:'Instrument Serif',Georgia,serif; font-size:3.1rem; line-height:1.05; margin:.25rem 0 .55rem;}
.dp-answer .big.warn {color:#F2B48C;}
.dp-answer p {color:#D6DEEB; margin:0; font-size:1rem; line-height:1.55;}
.dp-answer p b {color:#fff; font-weight:500;}
.dp-kpi {background:#fff; border:1px solid #D3D9E3; border-radius:12px; padding:.75rem 1rem; height:100%;}
.dp-kpi .l {font-size:.82rem; color:#5A5E66;}
.dp-kpi .v {font-size:1.5rem; font-weight:500; color:#0E1A2F; line-height:1.25; margin:.1rem 0;}
.dp-kpi .c {font-size:.8rem; color:#5A5E66;} .dp-kpi .c.warn {color:#B4531A;}
.dp-h {font-weight:600; font-size:1.05rem; color:#0E1A2F; margin:0; padding-top:1.3rem;}
.dp-sub {font-size:.84rem; color:#5A5E66; margin:0 0 .4rem;}
.dp-try {font-size:.84rem; color:#3D4047; background:#E7EBF2; border-radius:10px; padding:.6rem .8rem; margin-top:.4rem;}
.dp-note {font-size:.78rem; color:#5A5E66; margin-top:.5rem;}
.dp-list {margin:.2rem 0 0; padding-left:1.1rem; color:#3D4047; font-size:.92rem; line-height:1.7;}
.dp-list b {color:#0E1A2F; font-weight:500;}
@media (max-width:700px){ h1 {font-size:2.1rem !important;} .dp-answer .big {font-size:2.4rem;} }
</style>"""


def esc(s) -> str:
    return html.escape(str(s))


def header(title: str, lead_html: str, sample: bool = True):
    st.title(title)
    chip = '<span class="dp-chip">SAMPLE DATA</span>' if sample else ""
    st.markdown(f'<p class="dp-lead">{lead_html}{chip}</p>', unsafe_allow_html=True)


def step(label: str, where=st):
    where.markdown(f'<p class="dp-step">{esc(label)}</p>', unsafe_allow_html=True)


def answer(label: str, big: str, text_html: str, warn=False, where=st):
    """The one thing the user came for: a big number or verdict, and one sentence that explains it."""
    where.markdown(f'<div class="dp-answer"><span class="k">{esc(label)}</span>'
                   f'<div class="big{" warn" if warn else ""}">{esc(big)}</div><p>{text_html}</p></div>',
                   unsafe_allow_html=True)


def kpi(col, label, value, context="", warn=False):
    col.markdown(f'<div class="dp-kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
                 f'<div class="c{" warn" if warn else ""}">{"⚠ " if warn else ""}{esc(context)}</div></div>',
                 unsafe_allow_html=True)


def heading(title: str, sub: str = "", where=st):
    where.markdown(f'<p class="dp-h">{esc(title)}</p>' + (f'<p class="dp-sub">{esc(sub)}</p>' if sub else ""),
                   unsafe_allow_html=True)


def try_hint(text_html: str, where=st):
    where.markdown(f'<div class="dp-try"><b>Try it:</b> {text_html}</div>', unsafe_allow_html=True)


def note(text: str, where=st):
    where.markdown(f'<div class="dp-note">{esc(text)}</div>', unsafe_allow_html=True)


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
        return "good", "Yes, clearly better", (f"Even the low end of the likely range is above the {m} that matters, "
                                               f"so <b>{esc(other)}</b> is worth switching to.")
    if lo > 0 and hi >= mrd:
        return "good", "Yes, probably", (f"<b>{esc(other)}</b> is better, likely by a relevant amount, but the low "
                                         f"end of the range is below {m}. A few more measurements would settle it.")
    if lo > 0:
        return "small", "Better, but too little", (f"The difference is real, but the whole range is below the {m} "
                                                   "that matters. Not worth changing for this alone.")
    if hi < 0:
        return "worse", "No, it is worse", f"<b>{esc(other)}</b> measures lower than <b>{esc(base)}</b>."
    return "unclear", "Not proven yet", (f"The likely range includes zero, so the data cannot tell <b>{esc(other)}"
                                         f"</b> and <b>{esc(base)}</b> apart. More measurements would help.")


st.set_page_config(page_title="Measurement comparison (demo)", page_icon="📏", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

header("Did the change make a real difference?",
       "A greenhouse trial: plant height with a new fertiliser vs. the standard one. "
       "<b>Pick the two groups and the difference that would matter, get a yes, no or not yet.</b>",
       sample=st.session_state.get("upload") is None)

left, right = st.columns([1, 2], gap="large")
with left:
    step("1 · THE QUESTION")
    with st.container(border=True):
        up = st.session_state.get("upload")
        df = pd.read_csv(up if up else SAMPLE, sep=None, engine="python")
        num_cols = df.select_dtypes("number").columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols and 2 <= df[c].nunique() <= 20]
        if not num_cols or not cat_cols:
            st.error("Need at least one number column and one group column.")
            st.stop()
        main, own = st.container(), st.container()
        with own.expander("Your own data", expanded=bool(up)):
            st.file_uploader("CSV file (optional)", type=["csv"], key="upload",
                             help="One column with group names, one with the measured numbers.")
            value = st.selectbox("Measured value", num_cols, index=len(num_cols) - 1)
            group = st.selectbox("Group column", cat_cols)
            unit = st.text_input("Unit", "cm" if not up else "")
        with main:
            levels = df[group].dropna().unique().tolist()
            base = st.selectbox("Today (baseline)", levels, index=0)
            other = st.selectbox("New option", [g for g in levels if g != base])
            mrd = st.number_input(f"Worth it from a difference of ({unit or 'units'})", 0.0,
                                  value=2.0 if not up else 0.0, step=0.5,
                                  format="%.1f",
                                  help="Agree on this before looking at the result. It turns statistics into a decision.")
            with st.expander("Statistical settings"):
                level = st.select_slider("Confidence level", [0.90, 0.95, 0.99], value=0.95,
                                         format_func=lambda x: f"{x:.0%}")
                drop_out = st.checkbox("Exclude outliers (1.5 × IQR rule)", False)
    try_hint("set <b>Worth it from</b> to 1 or to 6 and see the answer change.", left)
    note("Runs in your browser. Your file is not uploaded anywhere and not stored.", left)

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
kind, big, text = verdict(lo, hi, mrd, unit, other, base)
u = f" {unit}" if unit else ""

extra = ""
if n_out and not drop_out:
    a2 = data.loc[(data[group] == base) & ~data["outlier"], value].to_numpy()
    b2 = data.loc[(data[group] == other) & ~data["outlier"], value].to_numpy()
    if len(a2) >= 3 and len(b2) >= 3:
        same = verdict(*analyse(a2, b2, level)["ci"], mrd, unit, other, base)[0] == kind
        extra = (f" {n_out} unusual measurement{'s' if n_out > 1 else ''} "
                 f"{'do' if n_out > 1 else 'does'} {'not ' if same else ''}change this answer.")

with right:
    step("2 · THE ANSWER")
    answer(f"IS {other.upper()} BETTER THAN {base.upper()}?", big, text + esc(extra), warn=kind != "good")
    k1, k2 = st.columns(2)
    kpi(k1, "Measured difference", f"{r['diff']:+.1f}{u}",
        f"{r['rel']:+.0%} · {b.mean():.1f} vs. {a.mean():.1f}{u} on average")
    kpi(k2, f"Likely range ({level:.0%})", f"{lo:+.1f} to {hi:+.1f}{u}",
        "includes zero: could be no effect" if lo <= 0 <= hi else
        (f"low end is below the {mrd:g}{u} that matters" if 0 < lo < mrd else f"{len(a)} vs. {len(b)} measurements"),
        warn=lo <= 0 <= hi or (0 < lo < mrd))

    heading("The likely difference against what matters",
            "Bar = likely range, dot = measured difference. Right of the dashed line = worth it.")
    span = max(abs(lo), abs(hi), mrd, 0.1) * 1.35
    iv = pd.DataFrame({"lo": [lo], "hi": [hi], "d": [r["diff"]], "y": ["difference"]})
    col = NAVY if kind == "good" else (WARN if kind in ("worse", "unclear") else GREY)
    xs = alt.Scale(domain=[min(-span * 0.4, lo * 1.2), span])
    rng = alt.Chart(iv).mark_rule(strokeWidth=12, color=col, opacity=0.35, strokeCap="round").encode(
        x=alt.X("lo:Q", scale=xs, title=f"{other} minus {base}{' (' + unit + ')' if unit else ''}",
                axis=alt.Axis(grid=False)), x2="hi:Q", y=alt.Y("y:N", axis=None))
    pt = alt.Chart(iv).mark_point(filled=True, size=170, color=col, opacity=1).encode(x="d:Q", y="y:N")
    lab = alt.Chart(iv).mark_text(dy=26, fontSize=13, color=INK, fontWeight=500).encode(
        x="d:Q", y="y:N", text=alt.Text("lab:N")).transform_calculate(
        lab=f"format(datum.d, '+.1f') + '{u}'")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GREY).encode(x="x:Q")
    zl = alt.Chart(pd.DataFrame({"x": [0], "t": ["no difference"]})).mark_text(
        align="right", dx=-5, dy=-46, color=MUTED, fontSize=12).encode(x="x:Q", text="t:N")
    layers = [zero, rng, pt, lab, zl]
    if mrd > 0:
        mdf = pd.DataFrame({"x": [mrd], "t": [f"worth it from {mrd:+g}{u}"]})
        layers += [alt.Chart(mdf).mark_rule(color=WARN, strokeDash=[5, 4]).encode(x="x:Q"),
                   alt.Chart(mdf).mark_text(align="left", dx=5, dy=-46, color=WARN, fontSize=12).encode(
                       x="x:Q", text="t:N")]
    st.altair_chart(chart_style(alt.layer(*layers).properties(height=140)), use_container_width=True)

    summary = used.groupby(group)[value].agg(["count", "mean", "std", "min", "median", "max"]).round(2)
    rows = [
        ("Test", "Welch t-test"),
        ("p-value", f"{r['p']:.4f}"),
        (f"{level:.0%} CI of difference (t)", f"{lo:.2f} to {hi:.2f}"),
        (f"{level:.0%} CI of difference (bootstrap)", f"{r['boot'][0]:.2f} to {r['boot'][1]:.2f}"),
        ("Effect size (Cohen's d)", f"{r['d']:.2f} ({size_word(r['d'])})"),
        ("Mann-Whitney U p-value", f"{r['p_mw']:.4f}"),
        ("Both groups roughly normal (Shapiro-Wilk)", "yes" if r["normal"] else "no, rely more on bootstrap"),
        ("Outliers found / excluded", f"{n_out} / {n_out if drop_out else 0}"),
        ("Worth it from", f"{mrd:g}{u}"),
    ]
    st.download_button("Download the report (Excel)",
                       excel_report(used.drop(columns="outlier"), summary,
                                    rows + [("Answer", big + ". " + re.sub("<[^>]+>", "", text))]),
                       file_name="comparison_report.xlsx", type="primary", use_container_width=True)

    with st.expander("See all measurements"):
        data["kind"] = np.where(data["outlier"], "unusual", np.where(data[group] == other, "new", "baseline"))
        pts = alt.Chart(data).mark_circle(size=55, opacity=0.8).encode(
            y=alt.Y(f"{group}:N", sort=[other, base], title=None, axis=alt.Axis(ticks=False, domain=False,
                                                                               labelPadding=8)),
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-1.5, 2.5])),
            x=alt.X(f"{value}:Q", scale=alt.Scale(zero=False), title=f"{value}{' (' + unit + ')' if unit else ''}"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["baseline", "new", "unusual"],
                                                      range=[GREY, NAVY, WARN]), legend=None),
            tooltip=list(data.columns.drop(["outlier", "kind"]))).transform_calculate(jitter="random()")
        mdf = used.groupby(group)[value].mean().reset_index()
        means = alt.Chart(mdf).mark_tick(color=INK, thickness=3, size=34).encode(
            y=alt.Y(f"{group}:N", sort=[other, base]), x=f"{value}:Q")
        st.altair_chart(chart_style((pts + means).properties(height=200)), use_container_width=True)
        st.caption("Each dot is one measurement, the dark tick is the group average"
                   + ("; orange = unusual value (outlier)." if n_out else "."))

    with st.expander("For the statistician: methods and checks"):
        st.dataframe(summary, use_container_width=True)
        st.dataframe(pd.DataFrame(rows, columns=["Check", "Value"]), hide_index=True, use_container_width=True)
        st.caption("This demo assumes independent measurements in two groups. In a real project the first step "
                   "is to check how the data were collected (randomisation, repeated measurements, batches).")
