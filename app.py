import streamlit as st
import intro
import eda_webapp
import ml_webapp
st.title("RapidInsights")
page = st.sidebar.selectbox("Choose a page", ["Data Exploration","Machine Learning","About the project"])
if page=='About the project':
    intro.hello()
if page=='Data Exploration':
    st.info('As a side note: There is an amazing ML platform to look at from the menu on the left')
    st.write('# **Streamlined EDA**')
    eda_webapp.eda_analysis()
if page=='Machine Learning':
    st.info('As a side note: There is an amazing EDA platform to look at from the menu on the left')
    st.write('# **Streamlined Machine Learning**')
    ml_webapp.ml_analysis()
    