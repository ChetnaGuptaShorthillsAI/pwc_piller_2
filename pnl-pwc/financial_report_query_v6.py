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
import tabula
from langchain.document_loaders import DirectoryLoader

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define a function to calculate the MD5 hash of a file's content
def calculate_hash(file_path):
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

# Path to the PDF file containing the financial report
pdf_path = 'pnl-pwc/axa_half_year_2023_financial_small.pdf'


# read PDF file
tables = tabula.read_pdf("pnl-pwc/axa_half_year_2023_financial_small.pdf", pages="all")

# save them in a folder
folder_name = "tables"
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
# iterate over extracted tables and export as excel individually
for i, table in enumerate(tables, start=1):
    table.to_excel(os.path.join(folder_name, f"table_{i}.xlsx"), index=False)


loader = DirectoryLoader('tables')    
documents_data = loader.load()

loader = PDFPlumberLoader(pdf_path)
documents_data1 = loader.load()
print("")

documents_data.extend(documents_data1)

# Embedding model settings
model_name = "BAAI/bge-large-zh"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
persist_directory = 'db/pdf_query'
collection_name = calculate_hash(pdf_path)
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
client = chromadb.PersistentClient(persist_directory)
try:
    client.get_collection(collection_name) 
    print("The collection exists.")
except:
    print("The collection does not exist. \nCreating a vector store for it. ")
    # Create Chroma Vector Store
    vectorstore = Chroma.from_documents(
        documents=documents_data,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    vectorstore.persist()

vectordbstore = Chroma(persist_directory=persist_directory, 
                collection_name=collection_name,
                embedding_function=embedding)

# Initialize Language Model
llm = AzureChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_type=os.getenv("OPENAI_API_TYPE"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("DEPLOYMENT_NAME")
)

# Initialize QA Chain
# qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                        retriever=vectordbstore.as_retriever(search_kwargs={"k": 10}),
#                                        return_source_documents=True)

qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordbstore.as_retriever(
                    search_type="mmr", search_kwargs={"fetch_k": 10}
                ),
                return_source_documents=True,
            )
# Main Query Loop
while True:
    question = input("\nEnter the question: ")
    if question.lower() == 'exit':
        break
    result = qa_chain({"query": question})
    
    print("\nRAW AI answer: ", result['result'])
    print(result['source_documents'])

    # Extract key-value pairs from the question and answer
    llm = AzureChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_type=os.getenv("OPENAI_API_TYPE"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        deployment_name=os.getenv("DEPLOYMENT_NAME")
    )

    sys_message = SystemMessage(content='''For the given piece of text, extract the keys and values, where the key is a question asked topic and values will be numerical values having no text or strings and return in the specified format. For text with no values, give the original N/A. Make sure you convert the millions, billions into universal numerical format and lakhs, crores into Indian numerical units.
    
    Follow the given formatting
    {
    'key 1' : 'Numerical Value',
    'key 2' : 'N/A'
    }

    Example:
    ...
    ''')

    human_msg = HumanMessage(content=f'''
    Instruction: From the question and answer provided, extract the key and value from it. Make sure you follow the specified formatting.
        Question: {question},
        Answer: {result['result']}
    AI response:
    ''')
    all_messages = []
    all_messages.append(sys_message)
    all_messages.append(human_msg)
   
    final_output = llm(all_messages).content
    
    print("\nAI answer: ", final_output)
