import streamlit as st
import intro
import eda_webapp


page = st.sidebar.selectbox("Choose a page", ["Intro", "Data Exploration","Machine Learning <Pending>"])
if page=='Intro':
    intro.hello()
if page=='Data Exploration':
    st.write('## **Streamined EDA**')
    eda_webapp.eda_analysis()
if page=='Machine Learning <Pending>':
    st.write('## **Streamined Machine Learning**')
    st.header("I am waiting for an interesting package to get updated that should work for this scenario. Trust me, it will be amazing once this is perfected! :sparkles:")
