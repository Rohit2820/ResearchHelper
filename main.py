import warnings
warnings.filterwarnings("ignore")
import os
import streamlit as st
import pickle
import time
import langchain
import time
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()
urls = []
st.title("News Research Tool ")
st.sidebar.title("News Article Urls")
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button('Process Urls')
file_path  = "vector_index.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    ## loading the data from articles 

    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loader Started........")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n",",","."],
                                              chunk_size=1000,chunk_overlap=200)
    main_placeholder.text("Text Splitter Started........")
    docs = splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    vectorstore_openai.save_local("vectorstore")
    # print(vectorstore_openai)

query = main_placeholder.text_input("I am ready to answer your questions : ")
    

if query :
    vectorstore=FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0.7),retriever=vectorstore.as_retriever())
    result = chain({'question':query},return_only_outputs=True)
    answer = result['answer']
    source = result['sources']
    st.header("Here is the Answer : ")
    st.write(answer)
    st.header("Here is the source of this answer : ")
    st.write(source)

