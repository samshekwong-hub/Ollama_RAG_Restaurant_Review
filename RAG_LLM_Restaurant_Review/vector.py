from importlib.metadata import metadata

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("Data/realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# check if chromadb location exist
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# if the location is not exist, we need to prepare all of our data by convert into documents
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )

        ids.append(str(i))
        documents.append(document)

# initialize the vector_store if db_location is existed
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}     #search keywords arguments: look up 5 relevant reviews & pass to llm
)