import streamlit as st
import intro
import eda_webapp
import ml_webapp

page = st.sidebar.selectbox("Choose a page", ["Intro", "Data Exploration","Machine Learning"])
if page=='Intro':
    intro.hello()
if page=='Data Exploration':
    st.write('## **Streamined EDA**')
    eda_webapp.eda_analysis()
if page=='Machine Learning':
    st.write('## **Streamined Machine Learning**')
    ml_webapp.ml_analysis()
    