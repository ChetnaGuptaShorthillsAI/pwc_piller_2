# import PyPDF2
# import tabula

# # Open the PDF file using PyPDF2
# with open(r"axa_half_year_2023_financial_small.pdf", 'rb') as pdf_file:
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     num_pages = len(pdf_reader.pages)

#     df = []
    
#     for page_num in range(num_pages):
#         if page_num == 0:
#             df.append(tabula.read_pdf(r"axa_half_year_2023_financial_small.pdf", area=(530, 12.75, 790.5, 561), pages=page_num + 1))
#         else:
#             df.append(tabula.read_pdf(r"axa_half_year_2023_financial_small.pdf", pages=page_num + 1))

# # Now you can work with the 'df' list, which contains DataFrames from each page.

# print(df)
# print(type(df))
import tabula
import os
from langchain.document_loaders import DirectoryLoader
# Importing required libraries
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
# from langchain.document_loaders.pdf import PDFPlumberLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
import chromadb
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
import hashlib



# read PDF file
tables = tabula.read_pdf("axa_half_year_2023_financial_small.pdf", pages="all")

# save them in a folder
folder_name = "tables"
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
# iterate over extracted tables and export as excel individually
for i, table in enumerate(tables, start=1):
    table.to_excel(os.path.join(folder_name, f"table_{i}.xlsx"), index=False)


loader = DirectoryLoader('tables')    
docs = loader.load()
print("")