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

# Read the Excel file
df = pd.read_excel('pnl-pwc/amazon_titile_vs_golden_set.xlsx')

# Convert the DataFrame to a list of dictionaries
data = df.to_dict(orient='records')

# Create a list of dictionaries with the desired structure
result = []
for item in data:
    result.append({key: item[key] for key in item.keys()})

# Specify the output JSON file path
output_file_path = 'output.json'

# Save the JSON data to a file
with open(output_file_path, 'w') as json_file:
    json.dump(result, json_file, indent=2)

# Print a message indicating the successful save
# print(f"JSON data has been saved to {output_file_path}")

file_path='output.json'
data = json.loads(Path(file_path).read_text())
# pprint(data)
loader = JSONLoader(
        file_path='output.json',
        jq_schema='.[]',
        text_content=False)

data = loader.load()

embeddings = cf.HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

db = FAISS.from_documents(data, embeddings)

query = "What is the precision score glove for s.no 7"
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



