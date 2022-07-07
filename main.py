import streamlit as st
from PIL import Image



def app():
    # set page configurations
    st.set_page_config(
        page_title="Churn Model Demo",
        page_icon="👋"
    )

    st.sidebar.success("Select a demo above")


    
    st.title('Churn Model Demo on Telco Customers')

    image = Image.open('assets/customer-churn.jpg')

    st.markdown(
        """
        Using streamlit, an open-source app framework to demonstrate a churn model developed for telco customers.
        """
    )

    st.image(image, caption='Image from Unsplash')

    st.markdown(
        """
        ### Demo Components:
        - EDA
        - Logistics Regression

        ### Resources:
        - Check out the original data source in [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)
        - For more information on the code base, visit my [GitHub repository](https://github.com/leonswl/churn-model)
        - Image by Austin Distel from [Unsplash](https://unsplash.com/photos/744oGeqpxPQ)
        """
    )

    

if __name__ == '__main__':
    app()