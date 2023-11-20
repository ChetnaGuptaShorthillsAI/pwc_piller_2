# Importing required libraries
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
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
import config as cf
from pprint import pprint


# data_val = '''
#        colm 1 | colm 2 | colm 3 | colm 4 | colm 5 |
# row 1: Null | Null | June 30, | Null | January 1, |
# row 2: Notes | (in EURO million) | 2023 | 2022, | 2022, restated |
# row 3: Null | Null | Null | restated | NULL |
# row 4: Null | Goodwill | 17,356 | 17,754 | 17,167 |
# '''

data_val='''
       colm 1 | colm 2 | colm 3 | colm 4 | colm 5 |
row 1:Notes | (in EURO million)| June 30,|December 31, | January 1, |
row 2: Null |2023 | 2022 | restated	2022 | restated|
row 3: Null |Shareholders equity- Group share| 45,912 | 46,071|51,885 |
row 4: Null | of which Net income - Group share | 2,906 | 3,018| 3,702 |
'''


def get_headers_values(data):

    prompt_val = f'''Act as an excel expert, who can understand the data headers and values. 
    The unstructured data has 4 rows, but some of the rows have header names, while some can have the values/ Your task is to identify what can be the possible header name. It is also a posibility that a single header name is distributed accross different columns and rows. 
    
    unstructured data: {data}

    Return me the header names in an organized format. 
    
    <The output should be in the given format>
    {{'Header 1':'Name of the header', ........}}
'''

    output = cf.llm.predict(prompt_val)

    print(output)

    return None


get_headers_values(data=data_val)