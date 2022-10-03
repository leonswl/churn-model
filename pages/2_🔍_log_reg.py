import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from src.log_reg import log_model, plot_roc_curve, plot_feat_impt
from src.utility import OneHotEncoding, apply_smote, plot_cnf_matrix


def log_reg(data):
    """
    Function to render Log Regression app
    """

    st.set_page_config(
        page_title="Log Regression",
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
        st.header("1. Data")
        with st.expander("Columns"):
            select_attribute = st.multiselect( label="Select Predictor Variables",
                            options=list(data.iloc[:,:-1].columns))
            st.session_state['attribute'] = select_attribute # update session state

        st.header("2.Model")
        with st.expander("Regularisation"):
            select_c = st.slider("Select regularisation strength C",min_value=0.1, max_value=20.0,value=1.0,step=0.1)
            st.session_state['C'] = select_c

    ## END - Sidebar

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
    model_instance = log_model(X_train_smote,y_train_smote,X_test,C=select_c)
    model_instance.fit()
    y_pred = model_instance.predict()
    print(f"Fit Logistic Regression - SUCCESS")

    # END - Logistic Regression

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

    # Container for Confusion Matrix and ROC Curve
    with st.container():
        st.subheader("Model Evaluation")
        st.markdown("""
        To evaluate our logistic regression model, the performance metrics used here will be the confusion matrix and AUC scores. 
        """)
        st.write("Model Score: ",round(model_instance.model.score(X_test,y_test),2))
        st.text('Model Report:\n ' + classification_report(y_test,y_pred))

        row2_col1, row2_col2 = st.columns(2)
        
        # Confusion Matrix
        with row2_col1:
            st.subheader("Confusion Matrix")
            cnf_matrix = plot_cnf_matrix(y_test,y_pred)
            st.pyplot(cnf_matrix)
            st.write('Accuracy (or Classification Rate): ',round(metrics.accuracy_score(y_test, y_pred),2))
            st.write('Precision: ',round(metrics.precision_score(y_test, y_pred,pos_label='Yes'),2))
            st.write('Recall: ',round(metrics.recall_score(y_test, y_pred,pos_label='Yes'),2))
            

        # ROC Curve
        with row2_col2:
            st.subheader("ROC Curve")
            roc_curve = plot_roc_curve(X_test, y_test,model_instance.model)
            st.pyplot(roc_curve)
            st.markdown("""
            Our AUC score is 0.835. AUC Score of 1 represents perfec classifier, and 0.5 represents a worthless classifier. In this case, our AUC looks decent.
            """)

    # Feature Importance
    with st.container():
        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            st.subheader("Feature Importance")
            st.markdown("To better understand the performance of our model, we can investigate each individual feature. To do so, we’ll start by getting each individual feature’s coefficient score:")
            feat_impt = plot_feat_impt(model_instance.model,X_encoded)
            st.pyplot(feat_impt)
            with st.expander("What are feature importance used for?"):
                st.markdown("""
                Scores marked with a zero coefficient, or very near zero coefficient, indicate that the model found those features unimportant and essentially removed them from the model. Positive scores indicate a feature that predicts class 1 (“yes”). Negative scores indicate a feature that predicts class 2 (“no”).
                """)



if __name__ == '__main__':
    df = pd.read_parquet('data/preprocessed.parquet')
    print(f"Load preprocessed data - SUCCESS")

    if 'attribute' not in st.session_state:
        st.session_state['attribute'] = None
    
    if 'c' not in st.session_state:
        st.session_state['C'] = 1

    log_reg(df)