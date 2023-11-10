import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import mean_absolute_error

# --- Using CSS to hide the Streamlit Branding
def css(filename):
    with open(filename) as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
css('./styles/styles.css')

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


# --- Building the model
X = final_df[['Years of Experience', 'Job Title', 'Education Level']] # Top 3 features
Y = final_df[['Salary']] # Target output

# Splitting the dataset into train and test sets (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=28)

# Training the model
reg_model = RandomForestRegressor(n_estimators=200, random_state=0)
reg_model.fit(x_train, y_train)


# --- Performance Evaluation
# Predict the target values of the test set
y_predict = reg_model.predict(x_test)

mea = mean_absolute_error(y_test, y_predict)
random_frst_oob = RandomForestRegressor(oob_score=True)
random_frst_oob.fit(x_train, y_train)

x_exp = x_test['Years of Experience'].to_numpy()
actual_sal = y_test.to_numpy()
predicted_sal = y_predict.flatten()

comparison_data = pd.DataFrame({
    'Years of Experience': x_test['Years of Experience'].to_numpy(),
    'Actual Salary': y_test.to_numpy().flatten(),
    'Predicted Salary': y_predict.flatten()
})





##### --- Web app section
# --- Main content section
with st.container():
    st.title("Step by Step Process")

st.divider()

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
    st.write("In this step, I used *SelectKBest* to run a chi-squared statistical test and find the features that contributes the most in finding the correct output. The top 3 features are the features that will be used in the future steps.")

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

    """, language="python", line_numbers=True)

    st.caption("Output:")
    st.write("Based on the results, the top three features are: (1) Years of Experience, (2) Job Title, (3) Education Level.")
    st.dataframe(feat_scores.sort_values(by='Score'))

# Step 6
with st.container():
    st.subheader("Step 6: Building the Model")
    st.write("Building the Random Forest Regressor Model.")

    st.caption("Input:")
    st.code("""
    X = final_df[['Years of Experience', 'Job Title', 'Education Level']] # Top 3 features
    Y = final_df[['Salary']] # Target output

    # Splitting the dataset into train and test sets (80/20)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=28)

    # Training the model
    reg_model = RandomForestRegressor(n_estimators=200, random_state=0)
    reg_model.fit(x_train, y_train)
    """, language="python", line_numbers=True)


# Step 7
with st.container():
    st.subheader("Step 7: Evaluating the Model's Performance")
    st.write("Evaluating the how well the model performed.")
    st.write("""
    - **Quick Guide to the Regression Metrics**
            
        **MAE** - This metric measures the average difference between the predicted and actual values. A lower MAE indicates better performance.
             
        **Out-of-bag score** - represents an estimate of how well the model is likely to perform on new, unseen data without the need for a separate validation dataset (the closer to 1, the better).         
    """)

    st.caption("Input:")
    st.code("""
    from sklearn.metrics import mean_absolute_error

    # Predict the target values of the test set
    y_predict = reg_model.predict(x_test)

    MEA = mean_absolute_error(y_test, y_predict)
    random_forest_oob = RandomForestRegressor(oob_score=True)
    random_forest_oob.fit(x_train, y_train)

    print("Regression Metrics")
    print(f"Mean Absolute Error: {MEA}")
    print(f"Out-of-bag score: {random_forest_oob.oob_score_}")

    print()

    # Visualizing the predicted salary and actual salary using scatterplots
    print("Comparison between the predicted salary and actual salary.")
    plt.scatter(x_test['Years of Experience'], y_test, color="blue", label="Actual")
    plt.scatter(x_test['Years of Experience'], y_predict, color="red", label="Predicted")
    plt.title("Predicted Salary vs Actual Salary")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

    """, language="python", line_numbers=True)

    st.caption("Output:")
    tab1, tab2 = st.tabs(["Regression Metrics", "Predicted Salary vs Actual Salary"])
    with tab1:
        st.write("**Regression Metrics**")
        st.write(f"- Mean Absolute Error (MEA): {mea}")
        st.write(f"- Out-of-bag score (OOB): {random_frst_oob.oob_score_}")
    with tab2:
        st.write("Comparison between the predicted salary and actual salary.")
        # Display the scatter chart
        st.scatter_chart(data=comparison_data, x="Years of Experience", y=["Actual Salary", "Predicted Salary"])