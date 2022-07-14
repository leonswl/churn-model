import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from src.log_reg import log_model, plot_cnf_matrix, plot_roc_curve, plot_feat_impt, OneHotEncoding, apply_smote
import matplotlib.pyplot as plt
from PIL import Image
import pickle


def log_reg(data):
    """
    Function to render Log Regression app
    """

    st.set_page_config(
        page_title="EDA",
        page_icon="📈",
        layout="wide"
    )

    st.markdown("# Churn Model using Logistics Regression Demo")
    
    st.write("""
        This demo illustrates the use of logistic regression to build a churn model for predicting churn outcomes. 
    """)


    ## START - Sidebar
    st.sidebar.header("Log Regression Demo")
    with st.sidebar:
        select_attribute = st.multiselect( label="Select Attributes as the Predictor Variables",
                        options=list(data.iloc[:,:-1].columns))
        st.session_state['attribute'] = select_attribute # update session state

    ## START - Sidebar

    ## Filter data based on User Input
    # split predictor and target variables to X and y respectively
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    if st.session_state['attribute'] == (None or []):
        filtered_X = X.copy() # data remains unchanged
    
    else:
        filtered_X = X[[c for c in df.columns if c in select_attribute]]

    # START - Run logistic regression model
    ##### ------  1. Perform Encoding of Categorical Variables ------ #####
    X_encoded = OneHotEncoding(filtered_X)
    print(f"Encoding Categorical Variables - SUCCESS")

    ##### ------  2. Split train and test set ------ #####
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

    ##### ------  3. Resolve Class Imbalance ------ #####
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    print(f"Balance Class - SUCCESS")


    ##### ------ 4. Fit Logistic Regression model ------ #####
    model_instance = log_model(X_train_smote,y_train_smote,X_test)
    model_instance.fit()
    y_pred = model_instance.predict()
    print(f"Fit Logistic Regression - SUCCESS")

    # END - Logistic Regression 

    # load saved model
    with open('data/model.pkl' , 'rb') as f:
        model_instance = pickle.load(f)

    # load model evaluation images
    cnf_matrix = Image.open('assets/cnf_matrix.png')
    roc_curve = Image.open('assets/roc_curve.png')
    feat_impt = Image.open('assets/feat_impt.png')


    with st.container():
        st.markdown("""
        Before fitting a logistic regression model, there are 2 crucial transformations required: 
        1) Encoding Categorical Variables
        2) Dealing with Class Imbalance
        """)

        row1_col1, row1_col2 = st.columns(2)

        # Row 1, Col 1
        with row1_col1:
            st.markdown("""
            ### Encoding Categorical Variables
            Of the 19 attributes, 15 of them are categorical variables. From our EDA, we observe that these variables are nominal in nature with no natural order. Hence we can simply employ one-hot encoding to convert them to numerical values

            **What is One-Hot Encoding?**
            > One hot encoding is one method of converting data to prepare it for an algorithm and get a better prediction. With one-hot, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns. Each integer value is represented as a binary vector
            """)

            ohc = """for var in cat_vars:
    data_encoded = pd.get_dummies(data_encoded,columns=[var])"""
            with st.expander("See Code Explanation"):
                st.code(ohc,language='python')
                st.markdown("""
                View [code source](https://github.com/leonswl/churn-model/blob/4c15b925ccb98d75417c48602a71e3384a8e92a8/src/log_reg.py#L11) for more details.
                """)
            st.write(X_encoded.head())

        # Row 1, Col 2
        with row1_col2:
            st.markdown("""
            ### Dealing with Class Imbalance
            From the EDA, we observed that number of users that didn't churned exceeds those that did by quite a far bit. This issue of class imbalance has to be addressed before fitting the model.

            Failure to address this issue might lead to high accuracy in predicting the majority class (users that didn't churn) but fail to capture the minority class.

            I'll leverage on over-sampling as the approach to deal with class imbalance. The technique used here will be Synthetic Minority Oversampling Technique, or SMOTE, which generates synthetic data for the minority class. This is only applied for our train data.

            **What is SMOTE?**  
            >SMOTE (Synthetic Minority Oversampling Technique) works by randomly picking a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.
            """)

            smote = """smote_nc = SMOTE()
x_smote, y_smote = smote_nc.fit_resample(X.to_numpy(),y.to_numpy())
            """

            with st.expander("See Code Explanation"):
                st.code(smote, language='python')
                st.markdown("""View [code source](https://github.com/leonswl/churn-model/blob/4c15b925ccb98d75417c48602a71e3384a8e92a8/src/log_reg.py#L33) for more details.
                """)

            st.write(y_train_smote.value_counts().reset_index().rename({0:'Users'},axis=1))
            st.markdown("""The number of users in each class is now equal and balanced.""")

    with st.container():
        st.subheader("Model Evaluation")
        st.markdown("""
        To evaluate our logistic regression model, the performance metrics used here will be the confusion matrix and AUC scores. 
        """)

        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.subheader("Confusion Matrix")
            st.image(cnf_matrix, caption='Confusion Matrix')
            st.markdown("""
            **Accuracy**
            Our accuracy score (or Classification Rate) is 75.2%, considered as a reasonable good accuracy.

            **Precision**
            Precision measures how accurate the model is when making a prediction (ho often it is correct). Our precision score is 51.8%, which means that the Logistic Regression model successfully predict users are going to churn 52.9% of the time. 

            **Recall**
            Our recall score is 79.6%, which means the Logistic Regression model can identfy users who have churn 79.9% of the time.
            """)

            st.subheader("Feature Importance")
            st.markdown("To better understand the performance of our model, we can investigate each individual feature. To do so, we’ll start by getting each individual feature’s coefficient score:")
            st.image(feat_impt, caption='Feature Importance')
            with st.expander("What are feature importance used for?"):
                st.markdown("""
                Scores marked with a zero coefficient, or very near zero coefficient, indicate that the model found those features unimportant and essentially removed them from the model. Positive scores indicate a feature that predicts class 1 (“yes”). Negative scores indicate a feature that predicts class 2 (“no”).
                """)
        
        with row2_col2:
            st.subheader("ROC Curve")
            st.image(roc_curve, caption='ROC Curve')
            st.markdown("""
            Our AUC score is 0.835. AUC Score of 1 represents perfec classifier, and 0.5 represents a worthless classifier. In this case, our AUC looks decent.
            """)



if __name__ == '__main__':
    df = pd.read_parquet('data/preprocessed.parquet')
    print(f"Load preprocessed data - SUCCESS")

    if 'attribute' not in st.session_state:
        st.session_state['attribute'] = None

    log_reg(df)