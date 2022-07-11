import streamlit as st
from src.log_reg import log_model
import matplotlib.pyplot as plt
import pickle


def log_reg():
    """
    Function to render Log Regression app
    """

    st.set_page_config(
        page_title="EDA",
        page_icon="📈",
        layout="wide"
    )

    st.markdown("# Exploratory Data Analysis (EDA) Demo")
    st.sidebar.header("EDA Demo")
    st.write("""
        This demo illustrates the descriptive analysis of our data. Some of the plots are quite heavy so it might take a few minutes for the entire demo to be completely rendered. 
    """)

    # load saved model
    with open('data/model.pkl' , 'rb') as f:
        model_instance = pickle.load(f)
    
    

if __name__ == '__main__':
    log_reg()