import streamlit as st
from pandasai import SmartDataframe
import pandas as pd
import os
from langchain_groq import ChatGroq
from pandasai import Agent

os.environ["PANDASAI_API_KEY"] = st.secrets["pandasai"]["api_key"]

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    api_key=st.secrets["groq"]["api_key"] ,

)

# Load the CSV file into a DataFrame
df = pd.read_csv("Data.csv")
#data = SmartDataframe(df,name = 'Retail Order Management Dataset',description = 'This dataset contains information about retail orders, including order details, customer information, and product details.')

st.set_page_config(page_title="Pandas AI with ChatGroq", layout="wide")

st.title("Pandas AI with ChatGroq")

# Preview of the Data
df_preview = df.head()  # Display first few rows for preview
st.write("### Preview of the Data")
st.dataframe(df_preview)

# Convert DataFrame to SmartDataframe
#smart_df = SmartDataframe(df, config={"llm": llm,})
agent = Agent(df,config={"llm": llm, "open_charts":False,"save_charts":False})
agent.train(docs='if there is spelling mistake please find the most related values also dont generate any graphs')

# Input Box for Querying the Data
st.write("### Ask questions about your data")
user_query = st.text_area("Enter your query:", "")

if st.button("Run Query"):
    with st.spinner("Processing your query..."):
        try:
            # Use SmartDataframe to process the query
            
            rephrased_query = agent.rephrase_query(user_query)
            print("The rephrased query is", rephrased_query)
            result = agent.chat(rephrased_query)
            explain = agent.explain()
            st.success("Query processed successfully!")
            st.write("### Result")
            st.write(result)
            st.write("### Explanation")
            st.write(explain)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


