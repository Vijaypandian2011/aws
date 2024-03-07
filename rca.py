# pip install streamlit 
# pip install langchain 
# pip install openai 
# pip install beautifulsoup4 
# pip install python-dotenv 
# pip install chromadb
# pip install langchain-openai==0.0.8
# pip install langchain-community

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

os.environ['OPENAI_API_KEY'] = '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'

# Function to get the OpenAI API key from the user through a modal
def no_ingest_docs():
    persist_directory = './vector_data'
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store
    


 
def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
 
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
 
    embeddings = OpenAIEmbeddings()
    vector_store=Chroma.from_documents(document_chunks, embedding=embeddings,persist_directory='./vector_data')
    vector_store.persist()
    return vector_store
 
 

   
 
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever=vector_store.as_retriever()
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)  
    return retriever_chain
   
def get_conversational_rag_chain(retriever_chain):  
    llm = ChatOpenAI()  
    prompt = ChatPromptTemplate.from_messages([
      ("system", "As a Chat Bot specialized in Mongo DB and server incident root cause analysis, You are designed to provide accurate responses to user queries regarding Mongo DB and server incident root cause analysis using information stored in your knowledge base. Whenever you don't know answer, you will polietly ask user to add that data url into your knowledge base. Your responses will always free from hallucinations and assumptions, focusing solely on reliable and precise answers related to Mongo DB and server troubleshooting. Now, Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])  
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)  
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)
 
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
   
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })  
    return response['answer']
 
# app config
st.set_page_config(page_title="Mongo Test App")
st.title("Mongo Bot")

 
# sidebar
with st.sidebar:
    st.header("Knowledge base")
    st.write("Add a URL if you wish to feed new data into the knowledge base. The bot will use this data to answer questions.")
    website_url = st.text_input("URL")
 
    if website_url is None or website_url == "":
        st.info("Add a URL to Process New Data")
 
    if st.button("Process New Data"):
        with st.spinner("Processing..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)              
            st.success("Processed Successfully!")         
 
    # session state
if "chat_history" not in st.session_state:
    st.session_state.vector_store = no_ingest_docs()
    st.session_state.chat_history = [
        AIMessage(content="Hello! How can I assist you today?"),
    ]

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
