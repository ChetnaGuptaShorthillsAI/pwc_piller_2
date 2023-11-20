import os
import openai
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

 

load_dotenv()

# openai.api_type = "azure"
# openai.api_version = "2023-05-15"
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_base = os.getenv("OPENAI_API_BASE")


openai_api_key= os.getenv("OPENAI_API_KEY")
openai_api_base=os.getenv("OPENAI_API_BASE")
openai_api_type= "azure"
openai_api_version="2023-05-15"
deployment_name=os.getenv("DEPLOYMENT_NAME")

llm = AzureChatOpenAI(temperature=0.0, 
                        openai_api_key= os.getenv("OPENAI_API_KEY"),
                        openai_api_base=os.getenv("OPENAI_API_BASE"),
                        openai_api_type= os.getenv("OPENAI_API_TYPE"),
                        openai_api_version=os.getenv("OPENAI_API_VERSION"), 
                        deployment_name=os.getenv("DEPLOYMENT_NAME"))