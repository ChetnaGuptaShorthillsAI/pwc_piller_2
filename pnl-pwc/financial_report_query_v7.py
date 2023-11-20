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


# Define the folder path where your Excel files are located
folder_path = "pnl-pwc/tables"

header_list=[]
# Loop through the Excel files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):  # Check if the file is an Excel file
        file_path = os.path.join(folder_path, filename)
        print(filename)
        # Read the Excel file using pandas
        df = pd.read_excel(file_path)
        
        # Extract the first five rows, including the header
        first_five_rows = df.head(5)
        
        # Convert the data to a list of dictionaries
        data_dict_list = str(first_five_rows.to_dict(orient='records'))
        
        query='You are presented with a DataFrame excerpt from a consolidated financial statement. Your task is to extract the column headers based on the provided rows and list them in order. Do not provide any explanations, just a numbered list of the headers. Avoid repetition and consider possible merged cells.DataFrame Excerpt:{data_dict_list}'
        # pprint(query)
        prompt=PromptTemplate(
            template=query,
            input_variables=[data_dict_list]
        )
        chain=cf.LLMChain(prompt=prompt,llm=cf.llm)
        response=chain.run(data_dict_list=data_dict_list)
        pprint(response)

        print("")
# Now, data_list contains the extracted data from all Excel files in the folder
# Each element in data_list is a list of dictionaries (one for each Excel file)

print("")
