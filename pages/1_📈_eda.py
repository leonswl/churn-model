import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_churn(data):
    """
    Function to plot churn ratio
    
    Args:
        data [dataframe]: preprocessed dataframe

    Returns:
        fig [figure]: matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, sharey=False, figsize=(10,10))
    
    sns.histplot(data=data, x='Churn',ax=ax)
    return fig


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_crosstabs_cat_vars(data):
    """
    Function to plot crosstab bar charts for the categorical variables against churn.

    Args:
        data [dataframe]: preprocessed dataframe

    Returns:
        fig [figure]: matplotlib figure

    """
    cat_covariates = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    fig, axes = plt.subplots(4, 4, sharey=False, figsize=(30,30))

    for i in enumerate(cat_covariates[:4]):
        data.groupby(by=[i[1],'Churn']).size().unstack().plot(kind='bar',stacked=True,ax=axes[0,i[0]])

    for i in enumerate(cat_covariates[4:8]):
        data.groupby(by=[i[1],'Churn']).size().unstack().plot(kind='bar',stacked=True,ax=axes[1,i[0]])

    for i in enumerate(cat_covariates[8:12]):
        data.groupby(by=[i[1],'Churn']).size().unstack().plot(kind='bar',stacked=True,ax=axes[2,i[0]])

    for i in enumerate(cat_covariates[12:]):
        data.groupby(by=[i[1],'Churn']).size().unstack().plot(kind='bar',stacked=True,ax=axes[3,i[0]])

    return fig

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_crosstabs_cont_vars(data):
    """
    Function to plot crosstab bar charts for the continuous variables against churn.

    Args:
        data [dataframe]: preprocessed dataframe

    Returns:
        fig [figure]: matplotlib figure    
    """

    cont_covariates = ['tenure','MonthlyCharges','TotalCharges']

    fig, axes = plt.subplots(1,3, sharey=False, figsize=(20,7))

    for i in enumerate(cont_covariates):
        sns.histplot(data=data, x=data[i[1]], multiple="stack", hue=data['Churn'], kde=True, ax=axes[i[0]]);

    return fig


# cache data to avoid reloading data
@st.cache
def load_data():
    """
    Function to load data
    """
    data = pd.read_parquet('data/preprocessed.parquet')
    return data

# main function to render app
def eda():
    """
    Function to render streamlit app on EDA
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
    
    # load dataframe
    df = load_data()

    with st.container():
        st.markdown("""
        ## About the dataset

        Lets take a look at the header of the dataset.
        """)
        st.write(df.head())

        r1_c1, r1_c2, r1_c3 = st.columns(3)
    
        # first column
        with r1_c1:
            df_shape = np.shape(df)
            st.write("Shape of dataset:", df_shape[0], "rows,", df_shape[1], "columns")    

            st.markdown("""
            Churn predictions often have a classis case of class imbalance. To observe if this phenomenon exist within the dataset, I'll plot the ratio of churn outcomes, the dependent variable. 

            Since the number of users who did not churn far exceeds those that churned, it appears that we have class imbalance. 

            ** Problem with Class Imbalance**
            Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce errors.
            """)

        # second column
        with r1_c2:
            churn_fig = plot_churn(df)
            st.pyplot(churn_fig)

        # third column as empty

    # container for categorical variables plots
    with st.container():
        st.subheader("Categorical Variables")
        st.markdown("""
        In this section, I'll explore how our categorical variables varies with churn. 
        """)
        # plot crosstab bar chart for categorical variables
        cat_var_fig = plot_crosstabs_cat_vars(df)
        with st.expander("See Categorical Variables Crosstabs"):
            st.pyplot(cat_var_fig)

    with st.container():
        st.subheader("Continuous Variables")
        st.markdown("""
        In this section, I'll explore how our continuous variables varies with churn.      
        """)
        # plot crosstab bar chart for continuous variables
        cont_var_fig = plot_crosstabs_cont_vars(df)
        with st.expander("See Continuous Variables Crosstabs"):
            st.pyplot(cont_var_fig)    

    with st.container():
        st.subheader("Discussion Points")
        st.markdown("""
        Looking at the descriptive exploration of categorical and continuous independent variables against the dependent variable (churn), we can observe that most of the variables have substantial variable and hence are potentially good predictors of churn. We will proceed to use all of these features for our modelling.
        """)


if __name__ == "__main__":
    eda()