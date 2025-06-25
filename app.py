import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import text
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- MySQL Connection ---
def connectDatabase(username, port, host, password, database):
    encoded_password = quote_plus(password)
    mysql_uri = f"mysql+mysqlconnector://{username}:{encoded_password}@{host}:{port}/{database}"
    st.session_state.db = SQLDatabase.from_uri(mysql_uri)

# --- Clean SQL from markdown formatting ---
def cleanSQL(query):
    return query.replace("```sql", "").replace("```", "").strip()

# --- Run SQL query and get rows + columns ---
def runQuery(query):
    if not st.session_state.db:
        return "Please connect to database", []

    engine = st.session_state.db._engine
    cleaned_query = cleanSQL(query)

    with engine.connect() as conn:
        result = conn.execute(text(cleaned_query))
        if result.returns_rows:
            rows = result.fetchall()
            columns = result.keys()
            return rows, columns
        else:
            return result.rowcount, []

# --- Get DB Schema ---
def getDatabaseSchema():
    return st.session_state.db.get_table_info() if st.session_state.db else "Please connect to database"

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key)

# --- Generate SQL from question using schema and memory ---
def getQueryFromLLM(question):
    chat_history = "\n".join([f"{chat['role'].capitalize()}: {chat['content']}" for chat in st.session_state.chat])
    template = """
You are an expert SQL developer. Generate an SQL query based on the user's question, database schema, and conversation history.

--- Conversation History ---
{chat_history}

--- Database Schema ---
{schema}

Only output the SQL query. Do not explain anything.

User Question:
{question}

SQL Query:
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({
        "question": question,
        "schema": getDatabaseSchema(),
        "chat_history": chat_history
    })
    return response.content.strip()

# --- Convert result to natural response ---
def getResponseForQueryResult(question, query, result):
    template2 = """
You are a data assistant. Based on the database schema, SQL query, and query result, write a natural language answer.

--- Schema ---
{schema}

Question: {question}
SQL Query: {query}
Result: {result}

Natural Language Answer:
"""
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain2 = prompt2 | llm
    response = chain2.invoke({
        "question": question,
        "schema": getDatabaseSchema(),
        "query": cleanSQL(query),
        "result": result[:3] if isinstance(result, list) else result
    })
    return response.content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="💬 Chat with MySQL DB", page_icon="🧠", layout="wide")
st.title("💬 AI-Powered MySQL Assistant")
st.markdown("Type a natural language question below to interact with your MySQL database.")

# --- Chat input ---
question = st.chat_input("Ask your MySQL database anything", key="chat_input")

# --- Session State ---
if "chat" not in st.session_state:
    st.session_state.chat = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "chat_tables" not in st.session_state:
    st.session_state.chat_tables = []

# --- Process question ---
if question:
    if "db" not in st.session_state:
        st.error("Please connect to the database first.")
    else:
        try:
            query = getQueryFromLLM(question)
            result, columns = runQuery(query)
            response = getResponseForQueryResult(question, query, result)

            # Convert result to table
            if isinstance(result, list) and len(result) > 0:
                df = pd.DataFrame(result, columns=columns)
            elif isinstance(result, int):
                df = pd.DataFrame({"Message": [f"✨ Rows affected: {result}"]})
            elif isinstance(result, str):
                df = pd.DataFrame({"Message": [result]})
            else:
                df = pd.DataFrame({"Message": ["ℹ️ No results found."]})

            st.session_state.chat.append({
                "role": "assistant",
                "content": f"**💡 You asked:** {question}\n\n**💬 Answer:** {response}"
            })
            st.session_state.query_history.append(cleanSQL(query))
            st.session_state.chat_tables.append(df)

        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
            st.session_state.chat_tables.append(pd.DataFrame({"Error": [str(e)]}))

# --- Display chat messages ---
for i, chat in enumerate(st.session_state.chat):
    with st.chat_message(chat['role']):
        st.markdown(chat['content'])
        if chat['role'] == 'assistant':
            if i < len(st.session_state.query_history):
                with st.expander("🗌 Show SQL Query"):
                    st.code(st.session_state.query_history[i], language="sql")
            if i < len(st.session_state.chat_tables):
                st.dataframe(st.session_state.chat_tables[i], use_container_width=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛠️ MySQL Connection")
    st.text_input("📥 Host", key="db_host", value="e.g., 127.0.0.1")
    st.text_input("📶 Port", key="db_port", value="e.g., 3306")
    st.text_input("👤 Username", key="db_username", value="e.g., root")
    st.text_input("🔐 Password", key="db_password", type="password")
    st.text_input("💼 Database", key="db_database", placeholder="Enter your database name")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Connect", use_container_width=True):
            if st.session_state.db_database.strip() == "":
                st.error("Please enter a database name.")
            else:
                try:
                    connectDatabase(
                        username=st.session_state.db_username,
                        port=st.session_state.db_port,
                        host=st.session_state.db_host,
                        password=st.session_state.db_password,
                        database=st.session_state.db_database,
                    )
                    st.success("✅ Connected!")
                except Exception as e:
                    st.error(f"❌ {e}")

    with col2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat = []
            st.session_state.query_history = []
            st.session_state.chat_tables = []
            st.success("✨ History cleared!")

    
    col_download1, col_download2 = st.columns(2)
    with col_download1:
        st.download_button("🡇 SQL History", "\n".join(st.session_state.query_history), file_name="query_history.sql", use_container_width=True)
    with col_download2:
        chat_text = "\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in st.session_state.chat])
        st.download_button("🡇 Chat Log", chat_text, file_name="chat_history.txt", use_container_width=True)
