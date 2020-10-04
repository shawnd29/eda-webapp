import streamlit as st
import intro
import eda_webapp
import ml_webapp
st.title("RapidInsights")
st.info("Do take a look at the menu on the left for additional choices")
page = st.sidebar.selectbox("Choose a page", ["Data Exploration","Machine Learning","About the project"])
if page=='About the project':
    intro.hello()
if page=='Data Exploration':
    st.write('## **Streamlined EDA**')
    eda_webapp.eda_analysis()
if page=='Machine Learning':
    st.write('## **Streamlined Machine Learning**')
    ml_webapp.ml_analysis()
    