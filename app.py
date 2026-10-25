import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
st.write("Predict the price per unit area using property details.")

st.divider()
st.subheader("Enter Property Details")

with st.form("house_form"):
    age=st.number_input("House age (years)",min_value=0.0,max_value=100.0,value=15.0)
    distance=st.number_input("Distance to nearest MRT station",min_value=0.0,value=500.0)
    stores=st.number_input("Number of convenience stores",min_value=0,max_value=20,value=5)
    latitude=st.number_input("Latitude",value=24.98,format="%.6f")
    longitude=st.number_input("Longitude",value=121.54,format="%.6f")
    date=st.number_input("Transaction date (dataset format)",min_value=2012.0,max_value=2014.0,value=2013.0,format="%.3f")
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
    prediction=model.predict(input_df)[0]

    st.divider()
    st.subheader("Prediction Result")
    st.metric("Estimated price per unit area",f"{prediction:.2f}")
    st.caption("The dataset reports price per unit area, not the total property price. Predictions are estimates, not guaranteed market values.")
