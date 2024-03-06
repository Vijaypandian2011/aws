# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb
#new croma
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
 
os.environ['OPENAI_API_KEY'] = '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'
 
# load_dotenv()
 
def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
 
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
 
    embeddings = OpenAIEmbeddings()
    vector_store=Chroma.from_documents(document_chunks, embedding=embeddings,persist_directory='./vector_data_openAI')
    vector_store.persist()
    return vector_store
 
 
def no_ingest_docs():
    persist_directory = './vector_data_openAI'
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store
   
 
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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
      ("system", "You are a Chat Bot used to answer users based on the information you have in vector db about Mongo DB and you will correctly answer the users based on the knowledge base you have without any hallucinations and assumptions.Now, Answer the user's questions based on the below context:\n\n{context}"),
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
st.set_page_config(page_title="RCA bot ")
st.title("RCA BOT - TEST 2")
 
# sidebar
with st.sidebar:
    st.header("Knowledge base")
    website_url = st.text_input("URL")
 
    if website_url is None or website_url == "":
        st.info("Add a URL to Process New Data")
 
    if st.button("Process New Data"):
        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, How can I help you?"),
            ]
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)        
 
if st.button("Use existing Data"):
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = no_ingest_docs()    
 
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
