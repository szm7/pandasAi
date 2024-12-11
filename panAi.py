import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from pandasai import Agent

os.environ["PANDASAI_API_KEY"] = st.secrets["general"]["PANDASAI_API_KEY"]

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    api_key=st.secrets["general"]["GROQ_API_KEY"] ,

)

# Load the CSV file into a DataFrame
df = pd.read_csv("Data.csv",na_values=['NULL', 'N/A',''],keep_default_na=False)
#data = SmartDataframe(df,name = 'Retail Order Management Dataset',description = 'This dataset contains information about retail orders, including order details, customer information, and product details.')

st.set_page_config(page_title="Pandas AI with ChatGroq", layout="wide")

st.title("Pandas AI with ChatGroq")
df_preview = df.head() 
st.write("### Preview of the Data")
st.dataframe(df_preview)

#smart_df = SmartDataframe(df, config={"llm": llm,})
agent = Agent(df,config={"llm": llm, "open_charts":False,"save_charts":False})
agent.train(docs='if there is spelling mistake please find the most related values also dont generate any graphs')

st.write("### Ask questions about your data")
user_query = st.text_area("Enter your query:", "")

if st.button("Run Query"):
    with st.spinner("Processing your query..."):
        try:
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


