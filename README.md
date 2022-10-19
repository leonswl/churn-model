# Churn Model on Telco data

Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download)
Running on: [streamlit cloud](https://leonswl-churn-model-main-5fs093.streamlitapp.com/)

---

## Project Directory
- [data](data): all transformations of the dataset are stored here
- [notebooks](notebook): playground for explorations in the form of notebooks
- [dependencies](requirements.txt): dependencies for replicating this project
- [assets](assets): contains various form of outputs either for rendering the app, or plots saved as images.
- [pipeline](src): source scripts for executing model pipeline
- [pages](pages): pages for multi-page streamlit app
- [app](main.py): main script for streamlit app


## Installation & Usage
```
# Initialise environment
python3 -m venv .venv

# initialise virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run streamlit locally
streamlit run main.py

# Run python files as scripts
python3 -m src.log_reg # example
```




