import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2
import streamlit as st
from streamlit_lottie import st_lottie

# ASSETS
animation = "https://lottie.host/dedcb31b-6506-43a2-a505-bf43a8042b69/2Q9BXeIUuV.json"

# CONSTANT VARIABLES
DF = pd.read_csv('./jupyter_nb/salary_dataset.csv')     # Dataset
YOE = DF.loc[:,'Years of Experience'].to_numpy()        # Years of Experience
SAL = DF.loc[:,'Salary'].to_numpy()                     # Salary

##### --- Machine Learning section
# --- Preprocessing
df_dup = pd.read_csv('./jupyter_nb/salary_dataset.csv')  # Duplicate of the original dataset

# Function for assigning numeric representation to non-numeric columns
def to_numeric(col):
    DF[col] = pd.factorize(DF[col].to_numpy())[0]

# Assigning numeric representation to Gender, Education Level and each Job Title
to_numeric('Gender')
to_numeric('Education Level')
to_numeric('Job Title')

DF.dropna(inplace=True)  # Dropping the NaN values
final_df = DF.drop(['Age'], axis=1)  # Dropping the 'Age' column

# Create a DataFrame with the columns you want to plot
data = {'Years of Experience': YOE, 'Salary': SAL}
df_to_plot = pd.DataFrame(data)


# --- Feature selection



##### --- Web app section
# --- Setting the page configuration
st.set_page_config(
    page_title="Salary Predictor", 
    page_icon="", 
    layout="wide"
)

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

st.divider()

# --- Main content section
# Step 1
with st.container():
    st.subheader("Step 1: Importing Python libraries")
    st.caption("Input:")
    st.code("""
            
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
            
    """, language="python", line_numbers=True)
    
# Step 2
with st.container():
    st.subheader("Step 2: Read the dataset")
    st.write("I used pandas methods to have a brief overview of the dataset. In this case, I used **.head()** to read the first 10 records from the dataset. I also checked the unique values of *Education Level* and *Job Title*.")

    st.caption("Input:")
    st.code("""
            
    from IPython.display import display
    df1 = pd.read_csv('./salary_dataset.csv')
    display(df1.head(10))

    # Display the unique values of Education Level
    display(There are {len(df1['Education Level'].unique())} unique values.)

    # Display the unique values of Job Title
    display(There are {len(df1['Job Title'].unique())} unique values.)
            
    """, language="python", line_numbers=True)

    st.caption("Output:")
    # Creating tabs
    tab1, tab2, tab3 = st.tabs(["Records", "Education Levels", "Job Titles"])
    head_df = df_dup.head(10) # Getting the first 10 records

    with tab1:
        st.caption("First 10 records of the dataset")
        st.dataframe(head_df)
    with tab2:
        st.caption("Education Level Unique Values")
        st.write(f"- There are **{len(DF['Education Level'].unique())} unique values.**")
    with tab3:
        st.caption("Job Title Unique Values")
        st.write(f"- There are **{len(DF['Job Title'].unique())} unique values.**")

# Step 3
with st.container():
    st.subheader("Step 3: Basic Data Exploration")
    st.write("I used pandas methods to perform some basic data exploration and retrieve useful information from the dataset.")

    st.caption("Input:")
    st.code("""

    # Basic Infos about the dataset
    df1.info()

    # Summary statistics
    display(df1.describe())

    # Getting the column of years of experience and salary then converting it to NumPy array
    x = df1.loc[:,'Years of Experience'].to_numpy()
    y = df1.loc[:,'Salary'].to_numpy()

    # Visualizing the correlation between years of experience and salary using scatterplot
    plt.scatter(x, y)
    plt.title("Years of Experience vs Salary")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.grid(True)
    plt.show()
            
    """, language="python", line_numbers=True)

    st.caption("Output:")
    # Creating tabs
    tab1, tab2, tab3 = st.tabs(["Dataset info", "Summary Statistics", "Salary vs Years of Experience"])

    with tab1:
        st.caption("Basic information about the dataset")
        buffer = io.StringIO()
        df_dup.info(buf=buffer)

        # Get the printed information as a string
        info_text = buffer.getvalue()

        # Display the information
        st.text(info_text)

    with tab2:
        st.caption("Summary Statistics of numeric data")
        st.dataframe(df_dup.describe())
    with tab3:
        st.caption("Scatterplot of Salary vs Years of Experience")
        st.scatter_chart(df_to_plot, x="Years of Experience", y="Salary")

# Step 4
with st.container():
    st.subheader("Step 4: Data Preprocessing")
    st.write("This step involves converting non-numerical data to numeric, dropping NaN values, and dropping unnecessary columns.")

    st.caption("Input:")
    st.code("""

    # Function for assigning numeric representation to non-numeric columns
    def to_numeric(col):
        df1[col] = pd.factorize(df1[col].to_numpy())[0]

    # Assigning numeric representation to Gender, Education Level and each Job Title
    to_numeric('Gender')
    to_numeric('Education Level')
    to_numeric('Job Title')

    # Dropping the two (2) rows with nan values
    df1.dropna(inplace=True)

    # I dropped the 'age' column because I think it's not a necessary feature. After all, I already have 'years of experience' column.
    final_df = df1.drop(['Age'], axis=1)

    print("Final Dataset")
    display(final_df)
            
    """, language="python", line_numbers=True)
    
    st.caption("Output:")
    st.write("Final Dataset")
    st.dataframe(final_df)

# Step 5
with st.container():
    st.subheader("Step 5: Feature Selection")
    st.write("In this step, I used *SelectKBest* to run a chi-squared statistical test and find the features that contributes the most in finding the correct output.")

    st.caption("Input:")
    st.code("""
    from sklearn.feature_selection import SelectKBest, chi2

    # Selecting the input data (columns 1-4) and output data (salary column)
    x = final_df.iloc[:, 0:4]
    y = final_df.iloc[:, -1]

    # Selecting the top 3 features
    top_feats = SelectKBest(score_func=chi2, k=3)
    fit = top_feats.fit(x, y)

    # Creating a dataframe for the scores and columns
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(x.columns)

    # Combining the features and their corresponding scores in one df
    feat_scores = pd.concat([df_columns, df_scores], axis=1)
    feat_scores.columns =  ['Features', 'Score']
    feat_scores.sort_values(by='Score')

    """)

    st.caption("Output:")
    