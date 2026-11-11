import streamlit as st

st.set_page_config(page_title="Coffee Shop Demand App", layout="wide")

st.title("Coffee Shop Demand App")
st.write("Use the menu on the left to open a page.")
st.markdown(
    """
- **Billing**: make a bill. Every bill is saved to `data/live_bills.xlsx`.
- **Forecast**: see which products are expected to sell more (coming in the next step).
"""
)
