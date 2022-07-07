import streamlit as st
import pandas as pd

# cache data to avoid reloading data
@st.cache
def load_data():
    data = pd.read_parquet('data/preprocessed.parquet')
    return data


def app():
    # set page configurations
    st.set_page_config(
        page_title="Churn Model Demo",
        page_icon="👋"
    )

    st.sidebar.success("Select a component above")


    df = load_data()
    st.title('Churn Model Demo on Telco Customers')

    st.markdown(
        """
        Using streamlit, an open-source app framework to demonstrate a churn model developed for telco customers.
        - Check out the original data source in [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)
        """
    )

    st.write(df.head())

if __name__ == '__main__':
    app()