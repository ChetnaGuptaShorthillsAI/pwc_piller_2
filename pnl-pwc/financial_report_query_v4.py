# Importing required libraries
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



def calculate_hash(file_path):
    """
    Calculate the MD5 hash of a file's content.

    This function reads the content of the specified file in 8 KB chunks and
    computes the MD5 hash of the entire file.

    Args:
        file_path (str): The path to the file to be hashed.

    Returns:
        str: The MD5 hash of the file's content as a hexadecimal string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as file:
            while True:
                data = file.read(8192)  # Read in 8 KB chunks to save memory
                if not data:
                    break
                hasher.update(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    return hasher.hexdigest()


def extract_values(question,answer):
    llm = AzureChatOpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_type=os.getenv("OPENAI_API_TYPE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            deployment_name=os.getenv("DEPLOYMENT_NAME")
        )

    sys_message = SystemMessage(content='''For the given piece of text, extract the keys and values, where the key is question asked topic and values will be numerical values having no text or strings and return in the specified format. For text with no values give original N/A. Make sure you convert the millions,billions into universal numerical format and lakhs, crores into indian numerical units.
    
    Follow the given formatting
    <Formating>
    {
    'key 1' : 'Numerical Value',
    'key 2' : 'N/A'
    }
    
    <Example>
    Instruction: 
    Instruction: From the question and answer provided, extract the key and value from it. Make sure you follow the         specified formatting.
        Question: What is the cost of building this bridge?,
        Answer: The cost of building the bridge is 4.82 million
    AI response: {
    'Building Cost' : '4,820,000',
    }
    Instruction: From the question and answer provided, extract the key and value from it. Make sure you follow the         specified formatting.
        Question: What is the profit for the year 2023,
        Answer:  The profit for the year 2023 was ₹11,459 crores.
    AI response: {
    'profit' : '1,14,59,00,00,000',
    }
    Instruction: From the question and answer provided, extract the key and value from it. Make sure you follow the         specified formatting.
        Question: What is the revenuefor the year 2020,
        Answer:  The revenue for the year 2020 was 16,775.88 lakhs.
    AI response: {
    'revenue' : '1,67,75,88,000',
    }
    <Example_end>
     ''')

    human_msg = HumanMessage(content=f'''
    Instruction: From the question and answer provided, extract the key and value from it. Make sure you follow the         specified formatting.
        Question: {question},
        Answer:{answer}
    AI response:
    ''')
    all_messages = []
    all_messages.append(sys_message)
    all_messages.append(human_msg)
   
    final_output = llm(all_messages).content
        # if "no value" or "No value"
        
    return final_output

 



def financial_report_query(pdf_path):
    """
    Query the financial statements of a company from a given PDF.

    Args:
        pdf_path (str): Path to the PDF file containing the financial report.

    Returns:
        None: Outputs the answer to the user's query about the financial report.
    """
    # Load PDF into documents
   
    loader = PDFPlumberLoader(pdf_path)
    documents_data = loader.load()
    print("")
    # Record the end time
    # Embedding model settings
    model_name = "BAAI/bge-large-zh"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    persist_directory = 'db/pdf_query'
    collection_name=calculate_hash(pdf_path)
    embedding=HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    client = chromadb.PersistentClient(persist_directory)
    try:
        client.get_collection(collection_name) 
        print("The collection exists.")
    except:
        print("The collection does not exist. \nCreating vector store for it. ")
        # Create Chroma Vector Store
        vectorstore = Chroma.from_documents(
            documents=documents_data,
            embedding= embedding,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        vectorstore.persist()
        
    vectordbstore = Chroma(persist_directory=persist_directory, 
                      collection_name= collection_name,
                      embedding_function=embedding)
    # Initialize Language Model
    llm = AzureChatOpenAI(
        temperature=1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_type=os.getenv("OPENAI_API_TYPE"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        deployment_name=os.getenv("DEPLOYMENT_NAME")
    )
    # Initialize QA Chain
    # qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    qa_chain = RetrievalQA.from_chain_type(llm =llm,
                                           retriever=vectordbstore.as_retriever(search_kwargs={"k": 10}),
                                           return_source_documents=True)
    # qa_chain = RetrievalQA.from_chain_type(
    #             llm=llm,
    #             chain_type="stuff",
    #             retriever=vectordbstore.as_retriever(
    #                 search_type="mmr", search_kwargs={"fetch_k": 10}
    #             ),
    #             return_source_documents=True,
    #         )

    # Main Query Loop
    while True:
        question = input("\nEnter the question: ")
        if question.lower() == 'exit':
            break
        result = qa_chain({"query": question})
        
        print("\nRAW AI answer: ", result['result'])
        print(result['source_documents'])
        refined_answer = extract_values(question,result['result'])
        print("\nAI answer: ",refined_answer)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Start the query interface
    pdf_path = 'pnl-pwc/axa_half_year_2023_financial_small.pdf'
    financial_report_query(pdf_path)
    
    

    """
What is the total assets for the year 2023
What was the operating profit for the year 2023
WHat was the earning per share for both year?    
WHat was the earning per share for 2022 and 2023?    
what was the dividend payout for the year 2023?
    """