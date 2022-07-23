import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None},ttl=300)
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


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None},ttl=300)
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

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None},ttl=300)
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
@st.cache(ttl=300)
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

    # load dataframe
    df = load_data()

    st.markdown("# Exploratory Data Analysis (EDA) Demo")
    st.sidebar.header("EDA Demo")
    st.write("""
        This demo illustrates the descriptive analysis of our data. Some of the plots are quite heavy so it might take a few minutes for the entire demo to be completely rendered. 
    """)

    ## START - Sidebar
    # st.sidebar.header("Log Regression Demo")
    # with st.sidebar:
    #     st.header("1. Data")
    #     with st.expander("Columns"):
    #         select_attribute = st.multiselect( label="Select Predictor Variables",
    #                         options=list(df.iloc[:,:-1].columns))
    #         st.session_state['attribute'] = select_attribute # update session state

    ## END - Sidebar

    # ## Filter data based on User Input
    # if st.session_state['attribute'] == (None or []):
    #     filtered_X = X.copy() # data remains unchanged
    
    # else:
    #     filtered_X = X[[c for c in df.columns if c in select_attribute]]


    with st.container():
        st.markdown("""
        ## About the dataset

        Lets take a look at the header of the dataset.
        """)
        st.write(df.head())

        r1_c1, r1_c2 = st.columns([2,1])
    
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

    # container for Demographics
    with st.container():
        st.subheader("Demographics")
        st.markdown("""
        It would be important to understand the gender, age range, patner and dependent status of the customers
        """)

        r2_c1, r2_c2, r2_c3 = st.columns(3)

        # Gender
        with r2_c1:
            st.markdown("#### Gender Distribution")
            gender_fig, gender_ax = plt.subplots(figsize=(8,6))
            sns.histplot(data=df,x='gender',ax=gender_ax)
            gender_ax.set_title("Gender Distribution");
            st.pyplot(gender_fig)
            st.markdown("""
            #### Gender Distribution
            The ratio of female to male is largely similar
            """)

        # Senior Citizens
        with r2_c2:
            st.markdown("#### Senior Citizens")
            sc_fig, sc_ax = plt.subplots(figsize=(8,6))
            plt.pie(df.SeniorCitizen.value_counts(),labels=['No','Yes'],autopct='%.0f%%');
            sc_ax.set_title("Proportion of Senior Citizens");
            st.pyplot(sc_fig)
            st.markdown("""
            There are only 16% of customers who are senior citizens. Majority of the customers belong to age groups of younger adults.
            """)

        with r2_c3:
            st.markdown("#### Partners & Dependents")
            partner_fig, partner_ax = plt.subplots(figsize=(10,8))
            df.groupby(['Partner','Dependents']).size().unstack().plot(kind='bar',stacked=True,ax=partner_ax)
            partner_ax.set_title('% Customers with/without dependents based on whether they have a partner');
            st.pyplot(partner_fig)
            st.markdown("""
            Distribution of partners among the customers are largely similar . However, much more customers who have dependents have depedents 
            """)

    # container for customer account information   
    with st.container():
        st.subheader("Customers Account")

        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            st.markdown("#### Tenure")
            tenure_fig, tenure_ax = plt.subplots(figsize=(10,8))
            tenure_ax = sns.histplot(data=df, x='tenure',multiple='stack', hue='Contract', kde=True,bins=36)
            tenure_ax.set_xlabel("Tenure (Months)")
            tenure_ax.set_ylabel("# of Customers")
            tenure_ax.set_title("Distribution of Tenure for each Contract type (stacked)");
            st.pyplot(tenure_fig)

            st.markdown("""
            A huge proportion of the users have been with the telco company for a couple of months. There's also a sizeable number of customers who been with the company for more than 70 months. Their length of tenure could be due to the contracts that they have, which we can likewise observe.

            **Month-to-month**
            Customers who are under this contract have a much shorter tenure (less than 1 year). These users are more likely to change their telcos.

            **One year**
            Customers who are under a yearly contract do not have much variance in their tenure.

            **Two year**
            Customers who are under a 2-yearly contract have a much higher tenure (opposite skew compared to Month-to-month). These users are likely more prone to stick to a Telco company once they have identified and contracted one.
            """)

        with r3_c2:
            st.markdown("#### Contract Types")
            contract_fig, contract_ax = plt.subplots(figsize=(8,6))
            contract_ax = df.Contract.value_counts().plot(kind='bar',rot=0)
            contract_ax.set_title("# of Customers by Contract Type")
            contract_ax.set_ylabel("# of Customers");
            st.pyplot(contract_fig)
            st.markdown("""
            Month-to-month contracts are the most popular among customers, with yearly or 2-yearly contracts still reasonably popular.
            """)

    # container for services
    with st.container():
        st.subheader("Services")
        services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

        services_fig, services_axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))
        for i, item in enumerate(services):
            if i < 3:
                ax = df[item].value_counts().plot(kind = 'bar',ax=services_axes[i,0],rot = 0)
                
            elif i >=3 and i < 6:
                ax = df[item].value_counts().plot(kind = 'bar',ax=services_axes[i-3,1],rot = 0)
                
            elif i < 9:
                ax = df[item].value_counts().plot(kind = 'bar',ax=services_axes[i-6,2],rot = 0)
            ax.set_title(item)
        with st.expander("Expand for more"):
            st.pyplot(services_fig)

    with st.container():
        st.subheader("Monthly vs Total Charges")
        st.markdown("""
            We observed that the total charges increases as the monthly bill for a customer increases, which supports a logical reasoning. 
            """)

        r4_c1, r4_c2 = st.columns(2)

        with r4_c1:
            
            charges_fig, charges_ax = plt.subplots(figsize=(10,8))
            sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Churn',ax=charges_ax)
            charges_ax.set_title("Monthly vs Total Charges broken down by Churn Outcomes");
            st.pyplot(charges_fig)
            st.markdown("""
            This pattern is mostly observed for customers who continued to stay with the Telco company. Customers who churned tend to have a higher monthly charges but low total charges.
            """)
            
        with r4_c2:
            charges1_fig, charges1_ax = plt.subplots(figsize=(10,8))
            sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Contract',ax=charges1_ax)
            charges1_ax.set_title("Monthly vs Total Charges broken down by Contract Type");
            st.pyplot(charges1_fig)
            st.markdown("""
            If we differentiate the customers by contracts, this pattern becomes even more distinct. Customers who are under yearly or 2-yearly contract have a linear relationship between monthly and total charges, whereas this is linearity isn't observed for customers on month-to-montht contract
            """)


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
    

    # if 'attribute' not in st.session_state:
    #     st.session_state['attribute'] = None

    eda()