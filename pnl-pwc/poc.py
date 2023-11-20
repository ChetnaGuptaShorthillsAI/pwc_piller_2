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

excel_file_path="pnl-pwc/Data Points.xlsx"
fs_file=pd.read_excel(excel_file_path, sheet_name='FS')
trial_balance_file=pd.read_excel(excel_file_path, sheet_name='Trial Balance')
# Convert the DataFrame to a list of dictionaries
fs_data = fs_file.to_dict(orient='records')
trail_balance_data=trial_balance_file.to_dict(orient='records')

# Create a list of dictionaries with the desired structure
result = []
for item in fs_data:
    result.append({key: item[key] for key in item.keys()})

for item in trail_balance_data:
    result.append({key: item[key] for key in item.keys()})
# Specify the output JSON file path
output_file_path = 'poc_output.json'

# Save the JSON data to a file
with open(output_file_path, 'w') as json_file:
    json.dump(result, json_file, indent=2)

# Print a message indicating the successful save
# print(f"JSON data has been saved to {output_file_path}")

file_path='poc_output.json'
data = json.loads(Path(file_path).read_text())
pprint(data)
loader = JSONLoader(
        file_path='poc_output.json',
        jq_schema='.[]',
        text_content=False)
data=loader.load()
embeddings = cf.HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

db = FAISS.from_documents(data, embeddings)

# query = "What is the precision score glove for s.no 7"
# docs = db.similarity_search(query)
qa_chain = RetrievalQA.from_chain_type(
                llm=cf.llm,
                chain_type="stuff",
                retriever=db.as_retriever(
                    search_type="mmr", search_kwargs={"fetch_k": 3}
                ),
                return_source_documents=True,
            )
while True:
    question = input("\nEnter the question: ")
    if question.lower() == 'exit':
        break
    result = qa_chain({"query": question})
    pprint(result['result'])

print("")



