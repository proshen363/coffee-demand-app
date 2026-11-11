from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

# ---------- settings ----------
BASE = Path(__file__).resolve().parent.parent
CATALOG_FILE = BASE / "data" / "product_catalog.xlsx"
LIVE_FILE = BASE / "data" / "live_bills.xlsx"

STORES = {"Astoria": 3, "Lower Manhattan": 5, "Hell's Kitchen": 8}
HISTORICAL_MAX_ID = 149456   # last transaction_id in the original sales file

COLUMNS = ["transaction_id", "transaction_date", "transaction_time", "transaction_qty",
           "store_id", "store_location", "product_id", "unit_price",
           "product_category", "product_type", "product_detail",
           "bill_id", "source", "entered_at"]


# ---------- helper functions ----------
@st.cache_data
def load_catalog():
    return pd.read_excel(CATALOG_FILE)


def read_live_bills():
    # returns everything saved so far (empty table if no bill yet)
    if LIVE_FILE.exists():
        return pd.read_excel(LIVE_FILE)
    return pd.DataFrame(columns=COLUMNS)


def save_bill(cart, store, source):
    # adds one row per item to live_bills.xlsx and returns a summary of the bill
    live = read_live_bills()
    next_id = max(HISTORICAL_MAX_ID, int(live["transaction_id"].max()) if len(live) else 0) + 1
    bill_no = live["bill_id"].nunique() + 1 if len(live) else 1
    bill_id = f"B{bill_no:04d}"
    now = datetime.now()

    if LIVE_FILE.exists():
        wb = load_workbook(LIVE_FILE)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "bills"
        ws.append(COLUMNS)
        for cell in ws[1]:
            cell.font = Font(bold=True)

    for i, item in enumerate(cart):
        ws.append([next_id + i, now.date(), now.time().replace(microsecond=0), item["qty"],
                   STORES[store], store, item["product_id"], item["price"],
                   item["category"], item["type"], item["detail"],
                   bill_id, source, now.strftime("%Y-%m-%d %H:%M:%S")])
        ws.cell(row=ws.max_row, column=2).number_format = "yyyy-mm-dd"
        ws.cell(row=ws.max_row, column=3).number_format = "hh:mm:ss"
    wb.save(LIVE_FILE)

    total = sum(item["qty"] * item["price"] for item in cart)
    return {"bill_id": bill_id, "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "store": store, "items": list(cart), "total": total}


# ---------- page ----------
st.set_page_config(page_title="Billing", layout="wide")
st.title("Billing")

catalog = load_catalog()

if "cart" not in st.session_state:
    st.session_state.cart = []
if "last_bill" not in st.session_state:
    st.session_state.last_bill = None

with st.sidebar:
    store = st.selectbox("Store", list(STORES))
    source = st.selectbox("Data type", ["demo", "live"],
                          help="demo = test bills, live = real sales. Demo rows can be left out when retraining.")

left, right = st.columns(2)

# left side: choose an item
with left:
    st.subheader("Add item")
    category = st.selectbox("Category", sorted(catalog["product_category"].unique()))
    in_category = catalog[catalog["product_category"] == category]
    ptype = st.selectbox("Product type", sorted(in_category["product_type"].unique()))
    in_type = in_category[in_category["product_type"] == ptype]
    detail = st.selectbox("Product", sorted(in_type["product_detail"].unique()))
    row = in_type[in_type["product_detail"] == detail].iloc[0]
    price = float(row["unit_price"])
    st.write(f"Price: **${price:.2f}**")
    qty = st.number_input("Quantity", min_value=1, max_value=20, value=1, step=1)

    if st.button("Add to bill"):
        st.session_state.cart.append({
            "product_id": int(row["product_id"]), "category": category, "type": ptype,
            "detail": detail, "price": price, "qty": int(qty)})

# right side: current bill
with right:
    st.subheader("Current bill")
    cart = st.session_state.cart
    if cart:
        table = pd.DataFrame(cart)
        table["line total"] = table["qty"] * table["price"]
        st.dataframe(table[["detail", "qty", "price", "line total"]], hide_index=True)
        st.metric("Bill total", f"${table['line total'].sum():.2f}")
    else:
        st.info("No items yet. Add an item on the left.")

    b1, b2 = st.columns(2)
    if b1.button("Remove last item", disabled=not cart):
        st.session_state.cart.pop()
        st.rerun()
    if b2.button("Clear bill", disabled=not cart):
        st.session_state.cart = []
        st.rerun()

    if st.button("Generate bill and save", type="primary", disabled=not cart):
        try:
            st.session_state.last_bill = save_bill(cart, store, source)
        except PermissionError:
            st.error("Could not save. Close live_bills.xlsx in Excel and press the button again.")
        else:
            st.session_state.cart = []
            st.rerun()

# receipt of the last saved bill
bill = st.session_state.last_bill
if bill:
    st.divider()
    st.success(f"Bill {bill['bill_id']} saved to data/live_bills.xlsx")
    st.write(f"{bill['store']}  |  {bill['time']}")
    receipt = pd.DataFrame(bill["items"])
    receipt["line total"] = receipt["qty"] * receipt["price"]
    st.dataframe(receipt[["detail", "qty", "price", "line total"]], hide_index=True)
    st.write(f"**Total: ${bill['total']:.2f}**")

# recently saved rows
st.divider()
st.subheader("Recently saved rows")
saved = read_live_bills()
if len(saved):
    st.dataframe(saved.tail(10).iloc[::-1], hide_index=True)
else:
    st.write("Nothing saved yet.")
