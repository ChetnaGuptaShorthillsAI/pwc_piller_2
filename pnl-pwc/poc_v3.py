import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
import json
from langchain.document_loaders import JSONLoader
from pathlib import Path
from pprint import pprint
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import config as cf

from langchain.schema import (
    HumanMessage,
    SystemMessage
)
import hashlib

import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders.pdf import PDFPlumberLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
import chromadb
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
import hashlib
import streamlit as st
from streamlit_chat import message


def extract_values(question,answer):
    llm = AzureChatOpenAI(
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_type=os.getenv("OPENAI_API_TYPE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            deployment_name=os.getenv("DEPLOYMENT_NAME")
        )

    prompt = f"""
    For the given data which is basically the question and answer from the ML output. Kindly clean the answer and show the output in readable and in crisp format. 
    Example:
    Question : Is the data has any security balance. 
    Answer: I'm Sorry, I am unable to find the information. 
    Response: NO
    
    Question: {question}
    Answer: {answer}
    Response:  
    
    """
    final_output = llm.predict(prompt)
    
    return final_output

def db_data():
    excel_file_path="Data Points.xlsx"
    fs_file=pd.read_excel(excel_file_path, sheet_name='FS')
    trial_balance_file=pd.read_excel(excel_file_path, sheet_name='Trial Balance')
    # Convert the DataFrame to a list of dictionaries
    fs_data = fs_file.to_dict(orient='records')
    trail_balance_data=trial_balance_file.to_dict(orient='records')

    # Create a list of dictionaries with the desired structure
    fs_result = []
    trial_balance_result = []
    for item in fs_data:
        fs_result.append({key: item[key] for key in item.keys()})
        
    fs_result= [{'source': 'financial statement', **item} for item in fs_data]

    for item in trail_balance_data:
        trial_balance_result.append({key: item[key] for key in item.keys()})
        
    trial_balance_result =  [{'source': 'trial_balance', **item} for item in trail_balance_data]
    # Specify the output JSON file path
    output_file_path = 'poc_output.json'
    result = fs_result + trial_balance_result
    # Save the JSON data to a file
    with open(output_file_path, 'w') as json_file:
        json.dump(result, json_file, indent=2)


    file_path='poc_output.json'
    data = json.loads(Path(file_path).read_text())
    # pprint(data)
    loader = JSONLoader(
            file_path='poc_output.json',
            jq_schema='.[]',
            text_content=False)
    data=loader.load()
    embeddings = cf.HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    db = FAISS.from_documents(data, embeddings)
    persist_directory = 'db'
    client = chromadb.PersistentClient(persist_directory)
    collection_name ="trial"
    try:
        client.get_collection(collection_name) 
        print("The collection exists.")
    except:
        print("The collection does not exist. \nCreating vector store for it. ")
        # Create Chroma Vector Store
        vectorstore = Chroma.from_documents(
            documents=data,
            embedding= embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        vectorstore.persist()
    vectordbstore = Chroma(persist_directory=persist_directory, 
                        collection_name= collection_name,
                        embedding_function=embeddings)
    return db 
    
def generate_response(question):
    
            
    db = db_data()
    # Initialize Language Model
    # query = "What is the precision score glove for s.no 7"
    # docs = db.similarity_search(query)
    qa_chain = RetrievalQA.from_chain_type(
                    llm=cf.llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(
                        search_type="mmr", search_kwargs={"fetch_k": 8}
                    ),
                    return_source_documents=True,
                )
    # qa_chain = RetrievalQA.from_chain_type(llm =llm,
    #                                            retriever=vectordbstore.as_retriever(search_kwargs={"k": 5}),
    #                                            return_source_documents=True)
    result = qa_chain({"query": question})
    # print("\nRAW AI answer: ", result['result'])
    # print(result['source_documents'])
    refined_answer = extract_values(question,result['result'])
    return refined_answer


def engine_0():
    st.title("Pillar 2 Demo")
    #storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    user_input=st.text_input("You:",key='input')
    if user_input:
        output=generate_response(user_input)
        #store the output
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')



engine_0()