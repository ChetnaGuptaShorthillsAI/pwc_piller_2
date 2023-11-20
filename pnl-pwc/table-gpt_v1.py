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
import config as cf

# # Read the Excel file
# df = pd.read_excel('pnl-pwc/amazon_titile_vs_golden_set.xlsx')

# # Convert the DataFrame to a list of dictionaries
# data = df.to_dict(orient='records')

# # Create a list of dictionaries with the desired structure
# result = []
# for item in data:
#     result.append({key: item[key] for key in item.keys()})

# # Specify the output JSON file path
# output_file_path = 'output.json'

# # Save the JSON data to a file
# with open(output_file_path, 'w') as json_file:
#     json.dump(result, json_file, indent=2)

# # Print a message indicating the successful save
# print(f"JSON data has been saved to {output_file_path}")

# file_path='output.json'
# data = json.loads(Path(file_path).read_text())
# pprint(data)
# for single_json in result:
loader = JSONLoader(
        file_path='test.json',
        jq_schema='.messages[].content',
        text_content=False)

data = loader.load()

embeddings = OpenAIEmbeddings()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = cf.RecursiveCharacterTextSplitter(
            separators=",",
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
docs = text_splitter.split_documents(data)

db = FAISS.from_documents(docs, embeddings)

query = "What is the precision score glove for s.no 7"
docs = db.similarity_search(query)
pprint(docs[0].page_content)




