import streamlit as st

from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# ASSETS
animation = "https://lottie.host/dedcb31b-6506-43a2-a505-bf43a8042b69/2Q9BXeIUuV.json"

# --- Setting the page configuration
st.set_page_config(
    page_title="Salary Predictor", 
    page_icon="", 
    layout="centered"
)

st.sidebar.success("Select a page above to navigate")

# --- Using CSS to hide the Streamlit Branding
def css(filename):
    with open(filename) as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
css('./styles/styles.css')

# --- Title section (brief introduction of the project)
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Salary Predictor :money_with_wings:")

        st.subheader("Short description")
        st.write("This project uses *Random Forest Regressor*— a machine learning regression algorithm— to predict the potential salary of a person. It uses a dataset that contains data such as  *Age, Gender, Education level, Job title, and Years of Experience*.")
        st.write("**NOTE:** The dataset used in this project was **large language models generated** and **not collected from actual data sources**. To learn more about the dataset, [click here](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer).")
    with right_column:
        st_lottie(animation, height=330, key="salary")