from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- settings ----------
BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
STORES = ["Astoria", "Hell's Kitchen", "Lower Manhattan"]
FIRST_DATE, LAST_DATE = date(2023, 1, 12), date(2023, 6, 29)   # dates covered by the sales history
OPEN_HOUR, LAST_HOUR = 6, 20                                   # shop hours seen in the data
OPTIONS = ["Next 2 hours", "Next 6 hours", "Next day (24 hours)"]
UP, DOWN = "#2E7D5B", "#B3402A"

CSS = """
<style>
.hero {background: linear-gradient(135deg, #2B1B14 0%, #6F4E37 55%, #B5651D 100%);
       padding: 24px 30px; border-radius: 18px; color: #fff; margin-bottom: 18px;
       box-shadow: 0 6px 18px rgba(43,27,20,.25);}
.hero-title {font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin: 0;}
.hero-sub {opacity: .85; margin: 4px 0 12px 0; font-size: 1rem;}
.pill {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
       padding: 4px 14px; border-radius: 999px; font-size: .82rem; margin: 0 8px 6px 0;}
.hero-note {margin-top: 6px; font-size: .85rem; color: #FFD9A8;}
.kpi {background: #FFFFFF; border: 1px solid #E7DDD3; border-left: 7px solid #B5651D; border-radius: 14px;
      padding: 14px 18px; box-shadow: 0 3px 10px rgba(43,27,20,.08); margin-bottom: 14px;}
.kpi-label {font-size: .74rem; font-weight: 700; color: #8C6A55; text-transform: uppercase; letter-spacing: .08em;}
.kpi-name {font-size: 1.1rem; font-weight: 700; color: #2B1B14; margin-top: 2px;}
.kpi-value {font-size: 2rem; font-weight: 800; line-height: 1.15;}
.kpi-sub {font-size: .82rem; color: #6B5A50;}
.panel-title {font-size: 1.25rem; font-weight: 800; color: #B5651D; margin: 0;}
.panel-sub {font-size: .85rem; color: #8C6A55; margin: 0 0 8px 0;}
button[data-variant="pills"][data-selected="true"] {background: #B5651D !important; border-color: #B5651D !important;}
button[data-variant="pills"][data-selected="true"] * {color: #FFFFFF !important; font-weight: 700;}
button[data-variant="pills"]:hover {border-color: #B5651D !important;}
</style>
"""


# ---------- loading ----------
@st.cache_resource(show_spinner="Loading models and sales history...")
def load_all():
    hourly_models = joblib.load(MODELS / "hourly_models.pkl")
    daily_model = joblib.load(MODELS / "daily_model.pkl")
    hourly = pd.read_pickle(MODELS / "features_hourly.pkl")
    daily = pd.read_pickle(MODELS / "features_daily.pkl")

    # "normal level" of each future window = sum of the usual hourly (or daily) sales in that window
    hourly = hourly.sort_values(["store_location", "product_type", "hour_bucket"]).reset_index(drop=True)
    g = hourly.groupby(["store_location", "product_type"], observed=True)["baseline_hourly"]
    hourly["normal_2h"] = sum(g.shift(-i) for i in range(1, 3))
    hourly["normal_6h"] = sum(g.shift(-i) for i in range(1, 7))

    daily = daily.sort_values(["store_location", "product_type", "date"]).reset_index(drop=True)
    daily["normal_1d"] = daily.groupby(["store_location", "product_type"], observed=True)["baseline_daily"].shift(-1)
    return hourly_models, daily_model, hourly, daily


# ---------- helper functions ----------
def system_reference(now):
    # hour to forecast from, based on the system clock.
    # The shop is closed before 6:00 and after 20:59, so a forecast made then starts from the
    # last open hour (20:00), which makes the forecast window begin at the next opening (6:00).
    ref = now.replace(minute=0, second=0, microsecond=0)
    note = ""
    if now.hour > LAST_HOUR:
        ref = ref.replace(hour=LAST_HOUR)
        note = "The shop is closed now, so the forecast starts from the next opening hour."
    elif now.hour < OPEN_HOUR:
        ref = (ref - timedelta(days=1)).replace(hour=LAST_HOUR)
        note = "The shop is closed now, so the forecast starts from the next opening hour."
    return ref, note


def latest_history_day(weekday):
    # most recent day in the saved history that falls on the same weekday
    d = LAST_DATE
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def whole(x):
    # round to a whole number (0.5 rounds up)
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def unit(x):
    return "unit" if int(whole(x)) == 1 else "units"


def make_table(products, expected, normal, actual):
    t = pd.DataFrame({"Product": products.astype(str).values,
                      "Expected": np.asarray(expected, dtype=float),
                      "Normal": np.asarray(normal, dtype=float),
                      "Actual": np.asarray(actual, dtype=float)})
    # normal level is floored at 1 unit so tiny numbers do not give huge percentages
    t["pct"] = (t["Expected"] - t["Normal"]) / t["Normal"].clip(lower=1) * 100
    t = t.dropna(subset=["pct"])
    # leave out products that sell almost nothing in this window, so they do not fill the top of the list
    return t[(t["Expected"] >= 0.5) | (t["Normal"] >= 0.5)]


def ranking_chart(top):
    d = pd.DataFrame({"product": top["Product"].values, "pct": top["pct"].round(0).values,
                      "expected": whole(top["Expected"]), "normal": whole(top["Normal"])})
    d["label"] = d["pct"].map(lambda v: f"{v:+.0f}%")
    d["dir"] = np.where(d["pct"] >= 0, "up", "down")
    lo, hi = min(0, d["pct"].min() * 1.35), max(10, d["pct"].max() * 1.6)

    base = alt.Chart(d).encode(
        y=alt.Y("product:N", sort=d["product"].tolist(), title=None,
                axis=alt.Axis(labelLimit=190, labelFontSize=13, ticks=False, domain=False)),
        x=alt.X("pct:Q", title=None, axis=None, scale=alt.Scale(domain=[lo, hi])))
    bars = base.mark_bar(cornerRadiusEnd=7, size=24).encode(
        color=alt.Color("dir:N", scale=alt.Scale(domain=["up", "down"], range=[UP, DOWN]), legend=None),
        tooltip=[alt.Tooltip("product:N", title="Product"), alt.Tooltip("expected:Q", title="Expected units", format=".0f"),
                 alt.Tooltip("normal:Q", title="Normal units", format=".0f"), alt.Tooltip("label:N", title="vs normal")])
    text_up = base.transform_filter("datum.pct >= 0").mark_text(align="left", dx=6, fontWeight="bold", fontSize=13, color=UP).encode(text="label:N")
    text_dn = base.transform_filter("datum.pct < 0").mark_text(align="right", dx=-6, fontWeight="bold", fontSize=13, color=DOWN).encode(text="label:N")
    return (bars + text_up + text_dn).properties(height=44 * len(d) + 10).configure_view(stroke=None)


def table_view(top, show_actual, compact=False):
    out = pd.DataFrame({"#": range(1, len(top) + 1),
                        "Product": top["Product"].values,
                        "Expected": whole(top["Expected"]).astype(int),
                        "Normal": whole(top["Normal"]).astype(int),
                        "Change": [f"{v:+.0f}%" for v in top["pct"]]})
    if show_actual and top["Actual"].notna().any():
        out["Actual"] = pd.Series(whole(top["Actual"]), index=out.index).astype("Int64")
    if compact:
        out = out[["Product", "Expected", "Change"]]
    return out


def panel(title, subtitle, t, top_n, show_actual, wide, compact=False):
    top = t.sort_values("pct", ascending=False).head(top_n)
    with st.container(border=True):
        st.markdown(f"<div class='panel-title'>{title}</div><div class='panel-sub'>{subtitle}</div>", unsafe_allow_html=True)
        if top.empty:
            st.info("No product has enough sales in this window.")
            return
        if wide:
            a, b = st.columns([1, 1])
            a.altair_chart(ranking_chart(top))
            b.dataframe(table_view(top, show_actual), hide_index=True)
        else:
            st.altair_chart(ranking_chart(top))
            st.dataframe(table_view(top, show_actual, compact), hide_index=True)


def window_text(k):
    # first and last hour covered by the next k open hours (closed hours overnight are skipped)
    one = hourly[(hourly["store_location"] == store) &
                 (hourly["product_type"] == rows_h["product_type"].iloc[0])]["hour_bucket"].reset_index(drop=True)
    i = one.index[one == ref_ts][0]
    start, end = one[i + 1], one[i + k] + pd.Timedelta(hours=1)
    return f"{start:%a %H:%M} to {end:%a %H:%M}" if start.date() != end.date() else f"{start:%a %H:%M} to {end:%H:%M}"


# ---------- page ----------
st.set_page_config(page_title="Forecast", page_icon="☕", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

hourly_models, daily_model, hourly, daily = load_all()

with st.sidebar:
    store = st.selectbox("Store", STORES)
    top_n = st.slider("Products to show", 3, 10, 5)
    st.button("Refresh time")
    manual = st.checkbox("Pick date and time manually", value=False)
    show_actual = False
    if manual:
        pick_date = st.date_input("Date", value=date(2023, 6, 24), min_value=FIRST_DATE, max_value=LAST_DATE)
        pick_hour = st.selectbox("Time", list(range(OPEN_HOUR, LAST_HOUR + 1)), index=3, format_func=lambda h: f"{h:02d}:00")
        show_actual = st.checkbox("Show actual sales from the data", value=True)

if manual:
    hist_date = pick_date
    ref_ts = pd.Timestamp(hist_date) + pd.Timedelta(hours=pick_hour)
    real_ref, note = ref_ts, ""
    time_pill = f"Chosen time: {ref_ts:%a %d %b %Y, %H:%M}"
    base_pill = "Taken from the sales history"
else:
    now = datetime.now()
    real_ref, note = system_reference(now)
    hist_date = latest_history_day(real_ref.weekday())
    ref_ts = pd.Timestamp(hist_date) + pd.Timedelta(hours=real_ref.hour)
    time_pill = f"System time: {now:%a %d %b %Y, %H:%M}"
    base_pill = f"Compared with {pd.Timestamp(hist_date):%a %d %b %Y} in the history"

note_html = f"<div class='hero-note'>{note}</div>" if note else ""
st.markdown(
    f"<div class='hero'><div class='hero-title'>Coffee Demand Forecast</div>"
    f"<div class='hero-sub'>Which products will sell more than usual</div>"
    f"<span class='pill'>Store: {store}</span><span class='pill'>{time_pill}</span>"
    f"<span class='pill'>{base_pill}</span>{note_html}</div>", unsafe_allow_html=True)

# rows for the chosen store and time
feats_h = hourly_models["target_2h"].feature_name_
feats_d = daily_model.feature_name_
rows_h = hourly[(hourly["store_location"] == store) & (hourly["hour_bucket"] == ref_ts)]
rows_d = daily[(daily["store_location"] == store) & (daily["date"] == pd.Timestamp(hist_date))]

if rows_h.empty or rows_h[feats_h].isna().any().any() or rows_d.empty or rows_d[feats_d].isna().any().any():
    st.warning("Not enough history for this date and time. Try a later date.")
    st.stop()

pred_2h = np.clip(hourly_models["target_2h"].predict(rows_h[feats_h]), 0, None)
pred_6h = np.clip(hourly_models["target_6h"].predict(rows_h[feats_h]), 0, None)
pred_1d = np.clip(daily_model.predict(rows_d[feats_d]), 0, None)

next_day = real_ref + timedelta(days=1)
forecasts = {
    OPTIONS[0]: (f"Window: {window_text(2)}", make_table(rows_h["product_type"], pred_2h, rows_h["normal_2h"], rows_h["target_2h"])),
    OPTIONS[1]: (f"Window: {window_text(6)}", make_table(rows_h["product_type"], pred_6h, rows_h["normal_6h"], rows_h["target_6h"])),
    OPTIONS[2]: (f"Day: {next_day:%A %d %b %Y}", make_table(rows_d["product_type"], pred_1d, rows_d["normal_1d"], rows_d["target_1d"])),
}

# choose which forecasts to show (one, two or all three)
chosen = st.pills("Show forecast", OPTIONS, selection_mode="multi", default=OPTIONS[:2], key="view")
chosen = [o for o in OPTIONS if o in (chosen or [])]
if not chosen:
    st.info("Choose at least one forecast above.")
    st.stop()

# headline cards: best product of each chosen forecast
cols = st.columns(len(chosen))
for col, name in zip(cols, chosen):
    t = forecasts[name][1].sort_values("pct", ascending=False)
    if t.empty:
        continue
    r = t.iloc[0]
    color = UP if r["pct"] >= 0 else DOWN
    col.markdown(
        f"<div class='kpi'><div class='kpi-label'>{name}: top product</div>"
        f"<div class='kpi-name'>{r['Product']}</div>"
        f"<div class='kpi-value' style='color:{color}'>{r['pct']:+.0f}%</div>"
        f"<div class='kpi-sub'>Expected {int(whole(r['Expected']))} {unit(r['Expected'])}, normal {int(whole(r['Normal']))}</div></div>",
        unsafe_allow_html=True)

# panels
if len(chosen) == 1:
    name = chosen[0]
    panel(name, forecasts[name][0], forecasts[name][1], top_n, show_actual, wide=True)
else:
    for col, name in zip(st.columns(len(chosen)), chosen):
        with col:
            panel(name, forecasts[name][0], forecasts[name][1], top_n, show_actual, wide=False, compact=len(chosen) >= 3)

with st.expander("How to read this"):
    st.write(
        "**Expected** is the number of units the model predicts will be sold in the window. "
        "**Normal** is the usual amount for the same store, product, weekday and hours in the past. "
        "**Change** is how far the expected amount is above (green) or below (red) normal. "
        "A product needs at least 1 normal unit to avoid tiny numbers giving huge percentages, "
        "and products that sell almost nothing in a window are left out. "
        "Units are shown as whole numbers. "
        "The next-day forecast is the total for the whole of tomorrow, not only the next 24 clock hours.")
