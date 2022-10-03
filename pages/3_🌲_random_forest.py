import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.utility import OneHotEncoding, apply_smote, plot_cnf_matrix
from sklearn.metrics import classification_report
from sklearn import metrics


def rand_forest(data):
    """
    Function to render Random Forest app
    """

    st.set_page_config(
        page_title="Random Forest",
        page_icon="🌲",
        layout="wide"
    )

    st.markdown("# Churn Model using Random Forest Demo")
    
    st.write("""
        This demo illustrates the use of Random Forest to build a churn model for predicting churn outcomes. 
    """)


    ## START - Sidebar
    st.sidebar.header("Random Forest Demo")
    with st.sidebar:
        st.header("1. Data")
        with st.expander("Columns"):
            select_attribute = st.multiselect( label="Select Predictor Variables",
                            options=list(data.iloc[:,:-1].columns))
            st.session_state['attribute'] = select_attribute # update session state

        # st.header("2.Model")
        # with st.expander("Regularisation"):
        #     select_c = st.slider("Select regularisation strength C",min_value=0.1, max_value=20.0,value=1.0,step=0.1)
        #     st.session_state['C'] = select_c

    ## END - Sidebar

    ## Filter data based on User Input
    # split predictor and target variables to X and y respectively
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    y = y.map(dict(Yes=1, No=0))

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
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(X_train_smote, y_train_smote)
    # predictions
    y_pred =forest_model.predict(X_test).round()


    # Container for Confusion Matrix and ROC Curve
    with st.container():
        st.subheader("Model Evaluation")
        st.markdown("""
        To evaluate our logistic regression model, the performance metrics used here will be the confusion matrix and AUC scores. 
        """)
        st.write("Model Score: ",round(forest_model.score(X_test,y_test),2))
        st.text('Model Report:\n ' + classification_report(y_test,y_pred))

        row2_col1, row2_col2 = st.columns(2)
        
        # Confusion Matrix
        with row2_col1:
            st.subheader("Confusion Matrix")
            cnf_matrix = plot_cnf_matrix(y_test,y_pred)
            st.pyplot(cnf_matrix)
            st.write('Accuracy (or Classification Rate): ',round(metrics.accuracy_score(y_test, y_pred),2))
            st.write('Precision: ',round(metrics.precision_score(y_test, y_pred),2))
            st.write('Recall: ',round(metrics.recall_score(y_test, y_pred),2))            

        # ROC Curve
        with row2_col2:
            st.subheader("ROC Curve")
            feat_importance = forest_model.coef_.flatten()
            st.write(feat_importance)
        #     roc_curve = plot_roc_curve(X_test, y_test,model_instance.model)
        #     st.pyplot(roc_curve)
        #     st.markdown("""
        #     Our AUC score is 0.835. AUC Score of 1 represents perfec classifier, and 0.5 represents a worthless classifier. In this case, our AUC looks decent.
        #     """)




    return


if __name__ == '__main__':
    df = pd.read_parquet('data/preprocessed.parquet')
    print(f"Load preprocessed data - SUCCESS")

    if 'attribute' not in st.session_state:
        st.session_state['attribute'] = None
    
    if 'c' not in st.session_state:
        st.session_state['C'] = 1

    rand_forest(df)
