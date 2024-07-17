#Import required libraries
import os 
from apikey import apikey 

import streamlit as st
import pandas as pd

# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

#OpenAIKey
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

# Load CSV file
file_path = "mergeddata.csv"  
df = pd.read_csv(file_path, low_memory=False)

#Title
st.title('AI Assistant 🤖')

#Welcoming message
st.write("Hello, 👋 I am your AI Assistant to answer your questions regarding Telangana growth analysis.")

#Explanation sidebar
with st.sidebar:
    st.write('*This application presents the Telangana growth analysis*')
    st.caption('''**This application presents the Exploratory Data Analysis (EDA) of Telangana datasets, offering insightful visualizations and statistical summaries to facilitate a comprehensive understanding of the data and its underlying patterns and trends. **
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> Telangana Data Exploration: Rich insights, dynamic visuals.</p>",unsafe_allow_html=True )

#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    # Display loaded DataFrame
    st.write("Data Overview")
    st.write("The first rows of your dataset look like this:")
    st.write(df.head())

    #llm model
    llm = OpenAI(temperature = 0)

    #Pandas agent
    pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    #Functions main
    @st.cache_data
    def function_agent():
        st.write("Data Overview")
        st.write("The first rows of your dataset look like this:")
        st.write(df.head())
        st.write("Data Cleaning")
        columns_df = pandas_agent.run("What are the meaning of the columns?")
        st.write(columns_df)
        missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
        st.write(missing_values)
        duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
        st.write(duplicates)
        st.write("Data Summarisation")
        st.write(df.describe())
        correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
        st.write(correlation_analysis)
        outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
        st.write(outliers)
        new_features = pandas_agent.run("What new features would be interesting to create?.")
        st.write(new_features)
        return

    @st.cache_data
    def function_question_variable():
        st.line_chart(df, y =[user_question_variable])
        summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
        st.write(summary_statistics)
        normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
        st.write(normality)
        outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
        st.write(outliers)
        trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
        st.write(trends)
        missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
        st.write(missing_values)
        return

    @st.cache_data
    def function_question_dataframe():
        dataframe_info = pandas_agent.run(user_question_dataframe)
        st.write(dataframe_info)
        return

    #Main
    st.header('Exploratory data analysis')
    st.subheader('General information about the dataset')

    with st.sidebar:
        with st.expander('What are the steps of EDA'):
            steps_eda = llm('What are the steps of EDA')
            st.write(steps_eda)

    function_agent()

    st.subheader('Variable of study')
    user_question_variable = st.text_input('What variable are you interested in')
    if user_question_variable is not None and user_question_variable !="":
        function_question_variable()

        st.subheader('Further study')

    if user_question_variable:
        user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
        if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
            function_question_dataframe()
        if user_question_dataframe in ("no", "No"):
            st.write("")