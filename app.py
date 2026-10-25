
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="House Price Prediction",page_icon="🏠",layout="centered")

@st.cache_resource
def load_model():
    df=pd.read_excel("Real estate valuation data set.xlsx")
    df=df.drop_duplicates().drop(columns=["No"],errors="ignore")
    target="Y house price of unit area"
    X=df.drop(columns=[target])
    y=df[target]
    model=make_pipeline(StandardScaler(),LinearRegression())
    model.fit(X,y)
    return model,X

model,X=load_model()

st.title("🏠 House Price Prediction")
st.write("Estimate property prices and convert them to Indian rupees.")

st.divider()
st.subheader("Property Details")

with st.form("house_form"):
    age=st.number_input("House Age (years)",min_value=0.0,max_value=100.0,value=15.0)
    distance=st.number_input("Distance to Nearest MRT Station",min_value=0.0,value=500.0)
    stores=st.number_input("Number of Convenience Stores",min_value=0,max_value=20,value=5)
    latitude=st.number_input("Latitude",value=24.98,format="%.6f")
    longitude=st.number_input("Longitude",value=121.54,format="%.6f")
    date=st.number_input("Transaction Date (dataset format)",min_value=2012.0,max_value=2014.0,value=2013.0,format="%.3f")

    st.divider()
    st.subheader("Price Conversion")

    exchange_rate=st.number_input(
        "Exchange rate (₹ per 1 TWD)",
        min_value=0.01,
        value=2.91,
        step=0.01,
        help="Example rate only. Update it to the current TWD-to-INR exchange rate."
    )

    area=st.number_input(
        "Property Area (square metres)",
        min_value=1.0,
        value=100.0,
        step=5.0
    )

    submitted=st.form_submit_button("Predict House Price",use_container_width=True)

if submitted:
    values={
        "X1 transaction date":date,
        "X2 house age":age,
        "X3 distance to the nearest MRT station":distance,
        "X4 number of convenience stores":stores,
        "X5 latitude":latitude,
        "X6 longitude":longitude
    }

    input_df=pd.DataFrame([values],columns=X.columns)
    prediction=float(model.predict(input_df)[0])

    # Dataset target is in units of 10,000 TWD per ping
    price_twd_per_ping=prediction*10000
    price_inr_per_ping=price_twd_per_ping*exchange_rate
    price_inr_per_m2=price_inr_per_ping/3.3058
    total_price=price_inr_per_m2*area

    st.divider()
    st.subheader("Estimated Property Value")

    col1,col2=st.columns(2)

    with col1:
        st.metric("Price per m²",f"₹{price_inr_per_m2:,.0f}")

    with col2:
        st.metric("Total Estimated Price",f"₹{total_price:,.0f}")

    st.subheader("Price Comparison")

    st.write(f"**Original prediction:** {prediction:.2f} units of 10,000 TWD per ping")
    st.write(f"**Price in Taiwan dollars:** NT$ {price_twd_per_ping:,.0f} per ping")
    st.write(f"**Converted price:** ₹{price_inr_per_ping:,.0f} per ping")
    st.write(f"**Area entered:** {area:,.2f} m²")

    st.caption(
        "The currency conversion uses the exchange rate you entered. "
        "The default rate is illustrative, not live. The total price is "
        "an estimate obtained by multiplying the converted unit price "
        "by your entered area. The model was trained on Taiwanese real "
        "estate data, so predictions for properties in other countries "
        "may not be reliable."
    )
