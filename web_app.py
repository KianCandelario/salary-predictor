import streamlit as st

st.set_page_config(page_title="", page_icon="", layout="wide")

# Title container (brief introduction of the project)
with st.container():
    st.title("Salary Predictor")

    st.subheader("Short description")
    st.write("This project uses *Random Forest Regressor*— a machine learning regression algorithm— to predict the potential salary of a person. It uses a dataset that contains data such as  *Age, Gender, Education level, Job title, and Years of Experience*.")
    st.write("**NOTE:** The dataset used in this project was **large language models generated** and **not collected from actual data sources**. To learn more about the dataset, [click here](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer).")

